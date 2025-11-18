#!/usr/bin/env python3
"""
Diagnostic script untuk troubleshoot import issues
Run this script dari notebooks/ directory untuk cek setup
"""
import sys
import os

print("=" * 60)
print("🔍 Laravel RAG LLM - Diagnostic Check")
print("=" * 60)

# 1. Check current directory
print(f"\n1. Current Directory:")
print(f"   {os.getcwd()}")

# 2. Check project structure
print(f"\n2. Expected Project Structure:")
notebook_dir = os.getcwd()

# Try to find project root
if os.path.basename(notebook_dir) == 'notebooks':
    project_root = os.path.dirname(notebook_dir)
else:
    project_root = notebook_dir

print(f"   Project root: {project_root}")

expected_dirs = ['src', 'data', 'configs', 'models', 'notebooks']
for d in expected_dirs:
    path = os.path.join(project_root, d)
    status = "✓" if os.path.exists(path) else "✗"
    print(f"   {status} {d}/ {'(exists)' if os.path.exists(path) else '(MISSING)'}")

# 3. Check src modules
print(f"\n3. Python Modules in src/:")
src_path = os.path.join(project_root, 'src')
if os.path.exists(src_path):
    py_files = [f for f in os.listdir(src_path) if f.endswith('.py')]
    for f in py_files:
        print(f"   ✓ {f}")
else:
    print(f"   ✗ src/ directory not found!")

# 4. Try adding to path and importing
print(f"\n4. Testing Imports:")
sys.path.insert(0, src_path)
print(f"   Added to sys.path: {src_path}")

modules_to_test = ['config_loader', 'data_processing', 'retrieval', 'model_utils']
for module in modules_to_test:
    try:
        __import__(module)
        print(f"   ✓ {module} - OK")
    except ImportError as e:
        print(f"   ✗ {module} - FAILED: {e}")

# 5. Check dependencies
print(f"\n5. Required Dependencies:")
dependencies = {
    'torch': 'PyTorch',
    'transformers': 'HuggingFace Transformers',
    'datasets': 'HuggingFace Datasets',
    'pandas': 'Pandas',
    'numpy': 'NumPy'
}

for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"   ✓ {name}")
    except ImportError:
        print(f"   ✗ {name} - NOT INSTALLED")

# 6. Summary
print(f"\n" + "=" * 60)
print("📋 SUMMARY & RECOMMENDATIONS")
print("=" * 60)

# Check if we're in the right directory
if os.path.basename(os.getcwd()) != 'notebooks':
    print("\n⚠️  WARNING: You're not in the notebooks/ directory!")
    print(f"   Current: {os.getcwd()}")
    print(f"   Expected: {os.path.join(project_root, 'notebooks')}")
    print("\n   FIX: cd into the notebooks/ directory first")

# Check if src exists
if not os.path.exists(src_path):
    print("\n❌ ERROR: src/ directory not found!")
    print("   Make sure you're running this from the project root or notebooks/")

# Check missing dependencies
missing_deps = []
for module in dependencies.keys():
    try:
        __import__(module)
    except ImportError:
        missing_deps.append(module)

if missing_deps:
    print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
    print(f"\n   FIX: Run this command:")
    print(f"   pip install {' '.join(missing_deps)}")

# Final message
if os.path.exists(src_path) and not missing_deps:
    print("\n✅ Everything looks good!")
    print("\n📝 To fix notebook imports, add this to the first cell:")
    print("""
import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))
    """)

print("\n" + "=" * 60)
