import uuid

server_address = "127.0.0.1:8188"
# server_address = "176.151.104.89:8188"
client_id = str(uuid.uuid4())

uri = f"ws://{server_address}/ws?clientId={client_id}"
