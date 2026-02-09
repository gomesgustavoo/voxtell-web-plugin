import pydoc
from queue import Queue
from threading import Thread
from typing import List, Tuple, Union

import numpy as np
import torch
from torch._dynamo import OptimizedModule
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import join, load_json

from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization
from nnunetv2.utilities.helpers import dummy_context, empty_cache

from voxtell.model.voxtell_model import VoxTellModel
from voxtell.utils.text_embedding import last_token_pool, wrap_with_instruction


class VoxTellPredictor:
    """
    Predictor for VoxTell segmentation model.
    
    This class handles loading the VoxTell model, preprocessing images,
    embedding text prompts, and performing sliding window inference to generate
    segmentation masks based on free-text anatomical descriptions.
    
    Attributes:
        device: PyTorch device for inference.
        network: The VoxTell model.
        tokenizer: Text tokenizer for prompt encoding.
        text_backbone: Text embedding model.
        patch_size: Patch size for sliding window inference.
        tile_step_size: Step size for sliding window (default: 0.5 = 50% overlap).
        perform_everything_on_device: Keep all tensors on device during inference.
        max_text_length: Maximum text prompt length in tokens.
    """
    def __init__(self, model_dir: str, device: torch.device = torch.device('cuda'),
                 text_encoding_model: str = 'Qwen/Qwen3-Embedding-4B') -> None:
        """
        Initialize the VoxTell predictor.
        
        Args:
            model_dir: Path to model directory containing plans.json and checkpoint.
            device: PyTorch device to use for inference (default: cuda).
            text_encoding_model: Pretrained text encoding model (Qwen/Qwen3-Embedding-4B).
            
        Raises:
            FileNotFoundError: If model files are not found.
            RuntimeError: If model loading fails.
        """
        # Device setup
        self.device = device
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        self.normalization = ZScoreNormalization(intensityproperties={})

        # Predictor settings
        self.tile_step_size = 0.5
        self.perform_everything_on_device = True

        # Embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoding_model, padding_side='left')
        self.text_backbone = AutoModel.from_pretrained(text_encoding_model, torch_dtype=torch.float16).eval()
        self.max_text_length = 8192

        # Load network settings
        plans = load_json(join(model_dir, 'plans.json'))
        arch_kwargs = plans['configurations']['3d_fullres']['architecture']['arch_kwargs']
        self.patch_size = plans['configurations']['3d_fullres']['patch_size']

        arch_kwargs = dict(**arch_kwargs)
        for required_import_key in plans['configurations']['3d_fullres']['architecture']['_kw_requires_import']:
            if arch_kwargs[required_import_key] is not None:
                arch_kwargs[required_import_key] = pydoc.locate(arch_kwargs[required_import_key])

        # Instantiate network
        network = VoxTellModel(
            input_channels=1,
            **arch_kwargs,
            decoder_layer=4,
            text_embedding_dim=2560,
            num_maskformer_stages=5,
            num_heads=32,
            query_dim=2048,
            project_to_decoder_hidden_dim=2048,
            deep_supervision=False
        )

        # Load weights
        checkpoint = torch.load(
            join(model_dir, 'fold_0', 'checkpoint_final.pth'),
            map_location=torch.device('cpu'),
            weights_only=False
        )

        if not isinstance(network, OptimizedModule):
            network.load_state_dict(checkpoint['network_weights'])
        else:
            network._orig_mod.load_state_dict(checkpoint['network_weights'])
        
        network.eval()
        self.network = network

    def preprocess(self, data: np.ndarray) -> Tuple[torch.Tensor, Tuple, Tuple[int, ...]]:
        """
        Preprocess a single image for inference.
        
        This function preprocesses an image already in RAS orientation by performing
        cropping to non-zero regions and z-score normalization.
        
        Args:
            data: Image data in RAS orientation (3D or 4D with channel dimension).
            
        Returns:
            Tuple containing:
                - Preprocessed image tensor
                - Bounding box of cropped region
                - Original image shape
        """

        if data.ndim == 3:
            data = data[None]  # add channel axis
        data = data.astype(np.float32)  # this creates a copy
        original_shape = data.shape[1:]
        data, _, bbox = crop_to_nonzero(data, None)
        data = self.normalization.run(data, None)
        data = torch.from_numpy(data)
        return data, bbox, original_shape
    
    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...]) -> List[Tuple]:
        """
        Generate sliding window slicers for patch-based inference.
        
        Args:
            image_size: Shape of the input image.
            
        Returns:
            List of slice tuples for extracting patches.
        """
        slicers = []
        if len(self.patch_size) < len(image_size):
            assert len(self.patch_size) == len(image_size) - 1, (
                'if tile_size has less entries than image_size, '
                'len(tile_size) must be one shorter than len(image_size) '
                '(only dimension discrepancy of 1 allowed).'
            )
            steps = compute_steps_for_sliding_window(image_size[1:], self.patch_size,
                                                     self.tile_step_size)
            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            tuple([slice(None), d, *[slice(si, si + ti) for si, ti in
                                                     zip((sx, sy), self.patch_size)]]))
        else:
            steps = compute_steps_for_sliding_window(image_size, self.patch_size,
                                                     self.tile_step_size)
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(
                            tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                  zip((sx, sy, sz), self.patch_size)]]))
        return slicers
    
    @torch.inference_mode()
    def embed_text_prompts(self, text_prompts: Union[List[str], str]) -> torch.Tensor:
        """
        Embed text prompts into vector representations.
        
        This function converts free-text anatomical descriptions into embeddings
        using the text backbone model.
        
        Args:
            text_prompts: Single text prompt or list of text prompts.
            
        Returns:
            Text embeddings tensor of shape (1, num_prompts, embedding_dim).
        """
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        n_prompts = len(text_prompts)
        self.text_backbone = self.text_backbone.to(self.device)

        text_prompts = wrap_with_instruction(text_prompts)
        text_tokens = self.tokenizer(
            text_prompts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        text_tokens = {k: v.to(self.device) for k, v in text_tokens.items()}
        text_embed = self.text_backbone(**text_tokens)
        embeddings = last_token_pool(text_embed.last_hidden_state, text_tokens['attention_mask'])
        embeddings = embeddings.view(1, n_prompts, -1)
        self.text_backbone = self.text_backbone.to('cpu')
        empty_cache(self.device)
        return embeddings

    @torch.inference_mode()
    def predict_sliding_window_return_logits(
        self,
        input_image: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform sliding window inference to generate segmentation logits.
        
        Args:
            input_image: Input image tensor of shape (C, X, Y, Z).
            text_embeddings: Text embeddings from embed_text_prompts.
            
        Returns:
            Predicted logits tensor.
            
        Raises:
            ValueError: If input_image is not 4D or not a torch.Tensor.
        """
        if not isinstance(input_image, torch.Tensor):
            raise ValueError(f"input_image must be a torch.Tensor, got {type(input_image)}")
        if input_image.ndim != 4:
            raise ValueError(
                f"input_image must be 4D (C, X, Y, Z), got shape {input_image.shape}"
            )
        
        self.network = self.network.to(self.device)

        empty_cache(self.device)
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():

            # if input_image is smaller than tile_size we need to pad it to tile_size.
            data, slicer_revert_padding = pad_nd_image(input_image, self.patch_size,
                                                       'constant', {'value': 0}, True, None)

            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

            predicted_logits = self._internal_predict_sliding_window_return_logits(
                data, text_embeddings, slicers, self.perform_everything_on_device
            )

            empty_cache(self.device)
            # Revert padding
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        return predicted_logits
    
    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(
        self,
        data: torch.Tensor,
        text_embeddings: torch.Tensor,
        slicers: List[Tuple],
        do_on_device: bool = True,
    ) -> torch.Tensor:
        """
        Internal method for sliding window prediction with Gaussian weighting.
        
        Uses a producer-consumer pattern with threading to overlap data loading
        and model inference.
        
        Args:
            data: Preprocessed image data.
            text_embeddings: Text embeddings for prompts.
            slicers: List of slice tuples for patch extraction.
            do_on_device: If True, keep all tensors on GPU during computation.
            
        Returns:
            Aggregated prediction logits.
            
        Raises:
            RuntimeError: If inf values are encountered in predictions.
        """
        results_device = self.device if do_on_device else torch.device('cpu')

        def producer(data_tensor, slicer_list, queue):
            """Producer thread that loads patches into queue."""
            for slicer in slicer_list:
                patch = torch.clone(
                    data_tensor[slicer][None],
                    memory_format=torch.contiguous_format
                ).to(self.device)
                queue.put((patch, slicer))
            queue.put('end')

        empty_cache(self.device)

        # move data to device
        data = data.to(results_device)
        queue = Queue(maxsize=2)
        t = Thread(target=producer, args=(data, slicers, queue))
        t.start()

        # preallocate arrays
        predicted_logits = torch.zeros((text_embeddings.shape[1], *data.shape[1:]),
                                        dtype=torch.half,
                                        device=results_device)
        n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

        gaussian = compute_gaussian(
            tuple(self.patch_size),
            sigma_scale=1. / 8,
            value_scaling_factor=10,
            device=results_device
        )

        with tqdm(desc=None, total=len(slicers)) as pbar:
            while True:
                item = queue.get()
                if item == 'end':
                    queue.task_done()
                    break
                patch, tile_slice = item
                prediction = self.network(patch, text_embeddings)[0].to(results_device)
                prediction *= gaussian
                predicted_logits[tile_slice] += prediction
                n_predictions[tile_slice[1:]] += gaussian
                queue.task_done()
                pbar.update()
        queue.join()

        # Normalize by number of predictions per voxel
        torch.div(predicted_logits, n_predictions, out=predicted_logits)
        
        # Check for inf values
        if torch.any(torch.isinf(predicted_logits)):
            raise RuntimeError(
                'Encountered inf in predicted array. Aborting... '
                'If this problem persists, reduce value_scaling_factor in '
                'compute_gaussian or increase the dtype of predicted_logits to fp32.'
            )
        return predicted_logits

    def predict_single_image(
        self,
        data: np.ndarray,
        text_prompts: Union[str, List[str]]
    ) -> np.ndarray:
        """
        Predict segmentation masks for a single image with text prompts.
        
        This is the main prediction method that orchestrates preprocessing,
        text embedding, sliding window inference, and postprocessing.
        
        Args:
            data: Image data in RAS orientation (3D or 4D with channel dimension).
            text_prompts: Single text prompt or list of text prompts describing
                anatomical structures to segment.
                
        Returns:
            Segmentation masks as numpy array of shape (num_prompts, X, Y, Z)
            with binary values (0 or 1) indicating the segmented regions.
        """

        # Preprocess image
        data, bbox, orig_shape = self.preprocess(data)

        # Embed text prompts
        embeddings = self.embed_text_prompts(text_prompts)

        # Predict segmentation logits
        prediction = self.predict_sliding_window_return_logits(data, embeddings).to('cpu')

        # Postprocess logits to get binary segmentation masks
        with torch.no_grad():
            prediction = torch.sigmoid(prediction.float()) > 0.5
        
        segmentation_reverted_cropping = np.zeros(
            [prediction.shape[0], *orig_shape],
            dtype=np.uint8
        )
        segmentation_reverted_cropping = insert_crop_into_image(
            segmentation_reverted_cropping, prediction, bbox
        )

        return segmentation_reverted_cropping


if __name__ == '__main__':
    from pathlib import Path
    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

    # Default paths - modify these as needed
    DEFAULT_IMAGE_PATH = "/path/to/your/image.nii.gz"
    DEFAULT_MODEL_DIR = "/path/to/your/model/directory"
    
    # Configuration
    image_path = DEFAULT_IMAGE_PATH
    model_dir = DEFAULT_MODEL_DIR
    text_prompts = ["liver", "right kidney", "left kidney", "spleen"]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load image
    img, props = NibabelIOWithReorient().read_images([image_path])
    
    # Initialize predictor and run inference
    predictor = VoxTellPredictor(model_dir=model_dir, device=device)
    voxtell_seg = predictor.predict_single_image(img, text_prompts)
    