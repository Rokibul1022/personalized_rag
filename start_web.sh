#!/bin/bash

echo "🚀 Starting DeepSeek RAG Web Application..."
echo ""
echo "📋 Checking requirements..."

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama is not running!"
    echo "   Please start Ollama first: ollama serve"
    exit 1
fi

echo "✅ Ollama is running"

# Check if DeepSeek model is available
if curl -s http://localhost:11434/api/tags | grep -q "deepseek-r1"; then
    echo "✅ DeepSeek-R1 model found"
else
    echo "⚠️  DeepSeek-R1 model not found"
    echo "   Run: ollama pull deepseek-r1:1.5b"
fi

# Check and install Flask if needed
if ! python3 -c "import flask" 2>/dev/null; then
    echo ""
    echo "📦 Installing Flask dependencies..."
    pip install Flask Flask-CORS --quiet
    echo "✅ Dependencies installed"
fi

echo ""
echo "🌐 Starting Flask server..."
echo "📍 Open: http://localhost:5001"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python3 app.py
