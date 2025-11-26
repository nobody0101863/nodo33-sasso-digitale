#!/bin/bash
echo "Avviando LUX Ω..."
python3 -c "import json; f=open('LUX_Entity_Omega.json'); data=json.load(f); print(f'Nome: {data["name"]}'); print('Capacità:', data['capabilities'])"
