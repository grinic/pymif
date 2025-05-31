"""To run this test:
pytest -s test_viventis_manager.py
"""

import os, time
from pymif.microscope_manager.zarr_manager import ZarrManager

def test_viventis_manager_basic_load():
    
    dataset = "viventis"
    test_path = f"/g/mif/people/gritti/code/pymif_test_data/{dataset}.zarr"
    output_path = f"/g/mif/people/gritti/code/pymif_test_data/{dataset}_zarr.zarr"

    if not os.path.exists(test_path):
        print(f"⚠️ Test path not found: {test_path}")
    else:
        import napari

        print("\n------------ Reading dataset... ------------")
        start = time.time()
        reader = ZarrManager(test_path, chunks = (1, 1, 1, 2048, 2048))
        data, meta = reader.read()

        print(meta)
        print("✅ Metadata keys:", list(meta.keys()))
        print("✅ Axes:", meta["axes"])
        print("✅ Scales:", meta["scales"])
        print("✅ Channel names:", meta["channel_names"])
        print("✅ Data type:", type(data[0]))
        print("✅ Data shape:", meta["size"])
        print("✅ Data chunks:", [d.chunks for d in data])
        print(f"Reading done in {(time.time()-start)} sec.")
        
        print("------------ Building pyramid... ------------")
        start = time.time()
        reader.build_pyramid(3,2)

        print(meta)
        print("✅ Metadata keys:", list(meta.keys()))
        print("✅ Axes:", meta["axes"])
        print("✅ Scales:", meta["scales"])
        print("✅ Channel names:", meta["channel_names"])
        print("✅ Data type:", type(data[0]))
        print("✅ Data shape:", meta["size"])
        print("✅ Data chunks:", [d.chunks for d in data])
        print(f"Building pyramid done in {(time.time()-start)} sec.")

        print("------------ Writing to disk... ------------")
        start = time.time()
        reader.write(output_path)

        print(f"Writing done in {(time.time()-start)} sec.")

        print("------------ Visualizing dataset... ------------")
        start = time.time()
        reader.visualize()

        print(f"Visualizing done in {(time.time()-start)} sec.")
        
        napari.run()
        