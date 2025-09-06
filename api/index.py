import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from main import app
from mangum import Mangum

# Vercel handler
handler = Mangum(app)