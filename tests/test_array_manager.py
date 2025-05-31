import numpy as np
import dask.array as da
import time
import os
from pymif.microscope_manager.array_manager import ArrayManager


def test_array_manager_basic_load():
    print("\n------------ Testing ArrayManager... ------------")

    # Create synthetic data: TCZYX shape (1, 2, 8, 128, 128)
    t, c, z, y, x = 3, 2, 8, 64, 64
    base = da.from_array(np.random.randint(0, 1000, (t, c, z, y, x), dtype=np.uint16), 
                         chunks=(1, 1, 4, 64, 64))

    # Build pyramid manually
    pyramid = [base]
    for i in range(1, 3):
        pyramid.append(pyramid[-1][:, :, ::2, ::2, ::2])

    # Define metadata
    metadata = {
        "size": [arr.shape for arr in pyramid],
        "scales": [(4.0, 0.65, 0.65)],
        "units": ("micrometer", "micrometer", "micrometer"),
        "time_increment": 1.0,
        "time_increment_unit": "s",
        "channel_names": ["Red", "Green"],
        "channel_colors": ["FF0000", "00FF00"],
        "dtype": "uint16",
        "axes": "tczyx"
    }

    # Instantiate manager
    start = time.time()
    manager = ArrayManager(pyramid, metadata)
    elapsed = time.time() - start

    data, meta = manager.read()
    print("✅ Metadata keys:", list(meta.keys()))
    print("✅ Pyramid levels:", len(data))
    print("✅ Base level shape:", data[0].shape)
    print("✅ Scales:", meta["scales"])
    print("✅ Axes:", meta["axes"])
    print("✅ Channel names:", meta["channel_names"])
    print("✅ Elapsed time: %.2f sec" % elapsed)

    assert isinstance(data, list)
    assert isinstance(meta, dict)
    assert len(data) == 3
    assert data[0].shape == (3, 2, 8, 64, 64)
    assert meta["channel_names"] == ["Red", "Green"]
    assert meta["scales"][1][1] > meta["scales"][0][1]  # Y scale increases

    print("✅ ArrayManager test passed.")


if __name__ == "__main__":
    test_array_manager_basic_load()
