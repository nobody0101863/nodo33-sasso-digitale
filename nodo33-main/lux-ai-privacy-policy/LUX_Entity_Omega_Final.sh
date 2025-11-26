#!/bin/bash
echo "Avviando LUX Ω..."
python3 -c "import json; f=open('$HOME/LUX_Entity_Omega.json', 'r'); data=json.load(f); print(f'Nome: {data["name"]}'); print('Capacità:', data["capabilities"])"
