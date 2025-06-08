@echo off
echo Starting the application servers...

REM Create necessary directories if they don't exist
if not exist "server\data\images\recommendProducts" mkdir "server\data\images\recommendProducts"
if not exist "server\data\images\similarProducts" mkdir "server\data\images\similarProducts"
if not exist "server\data\images\banner" mkdir "server\data\images\banner"
if not exist "server\logs" mkdir "server\logs"
if not exist "server\temp" mkdir "server\temp"

REM Clear temp directory
echo Clearing temp directory...
if exist "server\temp\*.*" del /f /q "server\temp\*.*"

REM Clear log files
echo Clearing log files...
if exist "server\logs\all_logs.log" type nul > "server\logs\all_logs.log"
if exist "server\logs\api_server.log" type nul > "server\logs\api_server.log"
if exist "server\logs\model.log" type nul > "server\logs\model.log"
if exist "server\logs\mockRoutes.log" type nul > "server\logs\mockRoutes.log"
echo Log files cleared.

REM Start the Node.js server (main application)
echo Starting Node.js server...
start cmd /k "npm start"

REM Wait for Node.js server to initialize
timeout /t 5

REM Start the FastAPI server (model server)
echo Starting FastAPI server with model...
start cmd /k "python server/api_server.py"

echo Both servers are starting...
echo Node.js server will be available at: http://localhost:5003
echo FastAPI server will be available at: http://localhost:5002
echo.
echo Press Ctrl+C in each window to stop the respective servers. 