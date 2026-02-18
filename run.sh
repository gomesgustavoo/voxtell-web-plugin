#!/usr/bin/env bash
# run.sh — Start VoxTell Web Interface (backend + frontend)
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

check_prerequisites() {
    local errors=0

    if ! command -v conda &>/dev/null; then
        for p in "$HOME/miniconda3/bin" "$HOME/anaconda3/bin"; do
            [ -f "$p/conda" ] && export PATH="$p:$PATH" && break
        done
        if ! command -v conda &>/dev/null; then
            echo -e "${RED}Error:${RESET} conda not found. Install Miniconda or Anaconda and run 'conda init bash'." >&2
            errors=$((errors + 1))
        fi
    fi

    if ! command -v npm &>/dev/null; then
        echo -e "${RED}Error:${RESET} npm not found. Install Node.js 20.x or higher." >&2
        echo "  Install via nvm: https://github.com/nvm-sh/nvm" >&2
        errors=$((errors + 1))
    elif command -v node &>/dev/null; then
        node_major=$(node --version 2>/dev/null | sed 's/v\([0-9]*\).*/\1/')
        if [ -n "$node_major" ] && [ "$node_major" -lt 20 ]; then
            echo -e "${RED}Error:${RESET} Node.js v${node_major} is too old — version 20.x or higher is required." >&2
            echo "  Update with nvm: nvm install --lts && nvm use --lts" >&2
            errors=$((errors + 1))
        fi
    fi

    if command -v conda &>/dev/null && ! conda env list 2>/dev/null | grep -q '^voxtell '; then
        echo -e "${RED}Error:${RESET} conda environment 'voxtell' not found." >&2
        echo "  Run the Installation steps in README.md first." >&2
        errors=$((errors + 1))
    fi

    if [ ! -f "$SCRIPT_DIR/frontend/node_modules/.bin/vite" ]; then
        echo -e "${RED}Error:${RESET} Frontend dependencies not installed." >&2
        echo "  Run: cd frontend && npm install" >&2
        errors=$((errors + 1))
    fi

    if [ ! -d "$SCRIPT_DIR/models/voxtell_v1.1" ]; then
        echo -e "${RED}Error:${RESET} Model not found at models/voxtell_v1.1/." >&2
        echo "  Run: python download_model.py" >&2
        errors=$((errors + 1))
    fi

    if [ "$errors" -gt 0 ]; then exit 1; fi
}

BACKEND_PID=""; FRONTEND_PID=""

cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${RESET}"
    for pid in "$FRONTEND_PID" "$BACKEND_PID"; do
        [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null || true
    done
    sleep 1
    for pid in "$FRONTEND_PID" "$BACKEND_PID"; do
        [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    done
    echo -e "${GREEN}Done.${RESET}"
}
trap cleanup EXIT INT TERM

check_prerequisites

echo ""
echo -e "${BOLD}VoxTell Web Interface${RESET}"
echo -e "─────────────────────────────────────────"

echo -e "${CYAN}Starting backend...${RESET}"
echo -e "  ${YELLOW}Note:${RESET} Model loading takes 30–60 s — inference won't work until you see 'Model loaded successfully.'"
conda run --no-capture-output -n voxtell python "$SCRIPT_DIR/backend/server.py" &
BACKEND_PID=$!

sleep 4

echo -e "${CYAN}Starting frontend...${RESET}"
(cd "$SCRIPT_DIR/frontend" && npm run dev 2>&1) &
FRONTEND_PID=$!

echo ""
echo -e "  Backend:  ${GREEN}http://localhost:8000${RESET}"
echo -e "  Frontend: ${GREEN}http://localhost:5173${RESET}  ${BOLD}← open this${RESET}"
echo ""
echo -e "  Press ${BOLD}Ctrl+C${RESET} to stop both servers."
echo -e "─────────────────────────────────────────"
echo ""

wait -n "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
