import asyncio
import websockets
import socket
import json
import time
import random

def generate_fake_data():
    entry = {
        "timestamp": time.time(),
        "bcg_signal": random.uniform(0, 1)
    }
    return json.dumps(entry)

# Define a handler function, which is called when a connection is established
async def echo(websocket, path):
    try:
        while True:
            # Send data 100 times per second
            for _ in range(100):
                await websocket.send(generate_fake_data())
                await asyncio.sleep(0.01)  # Wait for 0.01 seconds (100 times per second)
    except websockets.exceptions.ConnectionClosed:
        print("Connection closed")

# Get the local IP address
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

# Server configuration
local_ip = get_local_ip()
port = 8765

print(f"WebSocket server running on {local_ip}:{port}")

# Start the WebSocket server
start_server = websockets.serve(echo, local_ip, port)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
