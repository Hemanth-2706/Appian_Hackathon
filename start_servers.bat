@echo off
echo Starting the application servers...

REM Create necessary directories if they don't exist
if not exist "server\data\images\recommendProducts" mkdir "server\data\images\recommendProducts"
if not exist "server\data\images\similarProducts" mkdir "server\data\images\similarProducts"
if not exist "server\data\images\banner" mkdir "server\data\images\banner"

REM Start the Node.js server (main application)
echo Starting Node.js server...
start cmd /k "npm start"

REM Wait for Node.js server to initialize
timeout /t 5

REM Start the FastAPI server (model server)
echo Starting FastAPI server with model...
start cmd /k "python server/api_server.py"

echo Both servers are starting...
echo Node.js server will be available at: http://localhost:5000
echo FastAPI server will be available at: http://localhost:5001
echo.
echo Press Ctrl+C in each window to stop the respective servers. 