import http.server
import socketserver
import os
import webbrowser

os.chdir('frontend')

PORT = 3000
Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Frontend serving at http://localhost:{PORT}")
    webbrowser.open(f'http://localhost:{PORT}')
    httpd.serve_forever()