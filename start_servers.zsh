#!/usr/bin/env zsh
# ------------------------------------------------------------
# start_servers.zsh  –  macOS / zsh version of the .bat script
# ------------------------------------------------------------
set -e  # exit immediately on error

echo "Starting the application servers..."

# ------------------------------------------------------------
# 1. Create the required directories (idempotent with -p)
# ------------------------------------------------------------
mkdir -p server/data/images/recommendProducts
mkdir -p server/data/images/similarProducts
mkdir -p server/data/images/banner
mkdir -p server/logs
mkdir -p server/temp

# ------------------------------------------------------------
# 1a. Clear temp directory
# ------------------------------------------------------------
echo "Clearing temp directory..."
rm -f server/temp/* 2>/dev/null || true

# ------------------------------------------------------------
# 1b. Clear log files
# ------------------------------------------------------------
echo "Clearing log files..."
: > server/logs/all_logs.log
: > server/logs/api_server.log
: > server/logs/model.log
: > server/logs/mockRoutes.log
: > server/data/products.js
: > server/data/chatbot_g_n_a.js
echo "Log files cleared."

# ------------------------------------------------------------
# 2. Launch the Node.js server in a NEW Terminal window
# ------------------------------------------------------------
echo "Starting Node.js server..."
osascript <<EOF
tell application "Terminal"
    do script "cd \"$(pwd)\" && npm start"
end tell
EOF

# ------------------------------------------------------------
# 3. Give Node.js a moment to bind to port 5000
# ------------------------------------------------------------
sleep 5      # ⇔  timeout /t 5

# ------------------------------------------------------------
# 4. Launch the FastAPI server (model) in ANOTHER window
# ------------------------------------------------------------
echo "Starting FastAPI server with model..."
osascript <<EOF
tell application "Terminal"
    do script "cd \"$(pwd)\" && source venv/bin/activate && uvicorn server.api_server:app --reload --port 5001"
end tell
EOF

# ------------------------------------------------------------
# 5. Final messages – mirrors the .bat output
# ------------------------------------------------------------
echo "Both servers are starting..."
echo "Node.js server will be available at: http://localhost:5000"
echo "FastAPI server will be available at: http://localhost:5001"
echo
echo "Press Ctrl+C in each window to stop the respective servers."
