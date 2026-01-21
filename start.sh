#!/bin/bash
# SmartRoot-AI Quick Start Script

echo "╔════════════════════════════════════════════════════════════╗"
echo "║          🌱 SmartRoot-AI - Quick Start Guide 🌱           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if in correct directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please navigate to the project directory."
    echo "   cd /home/dinesh/Minor_project/smartroot_ai"
    exit 1
fi

echo "✅ Project directory verified"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python3 not found. Please install Python 3.8 or higher."
    exit 1
fi
echo "✅ Python3 found"
echo ""

# Check virtual environment
if [ ! -d "venv_tf" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv_tf
    source venv_tf/bin/activate
    pip install --upgrade pip
else
    echo "✅ Virtual environment exists"
    source venv_tf/bin/activate
fi
echo ""

# Run tests
echo "🧪 Running application tests..."
echo "────────────────────────────────────────────────────────────"
python3 test_application.py
TEST_RESULT=$?
echo ""

if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ All tests passed!"
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║         🚀 Starting SmartRoot-AI Application 🚀            ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📖 To stop the application, press Ctrl+C"
    echo ""
    echo "🌐 The app will be available at:"
    echo "   • Local: http://localhost:8501"
    echo "   • Remote: http://0.0.0.0:8501"
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo ""
    
    # Start the app (bind to all interfaces so mobile can access)
    streamlit run app.py --server.address 0.0.0.0 --server.port 8501
else
    echo "❌ Tests failed. Please check the errors above."
    echo ""
    echo "📖 Troubleshooting Tips:"
    echo "   • Verify all dependencies are installed"
    echo "   • Check that model/vetiver_cnn.h5 exists"
    echo "   • Ensure all Python files have correct syntax"
    echo "   • Review VERIFICATION_REPORT.md for details"
    exit 1
fi
