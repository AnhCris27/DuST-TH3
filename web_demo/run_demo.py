import os
import sys
import webbrowser
import uvicorn

if __name__ == '__main__':
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(demo_dir)
    sys.path.insert(0, os.path.abspath(os.path.join(demo_dir, '..')))
    
    port = 8000
    url = f"http://localhost:{port}"
    print(f"\n{'='*60}")
    print(f"  DUST: Dual Swin Transformer Multi-Modal AI Demo")
    print(f"  Server starting at: {url}")
    print(f"  API Docs (Swagger): {url}/docs")
    print(f"{'='*60}\n")
    
    uvicorn.run("app:app", host="127.0.0.1", port=port, log_level="info")
