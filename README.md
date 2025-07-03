# Create virtual environment in the root directory
python -m venv .venv

# Activate (each new terminal)
# macOS/Linux:
source .venv/bin/activate
# Windows CMD:
.venv\Scripts\activate.bat
# Windows PowerShell:
.venv\Scripts\Activate.ps1


# Install dependencies
pip install -r requirements.txt


# set OPENAI_API_BASE=http://localhost:11434/v1
# echo %OPENAI_API_BASE%

