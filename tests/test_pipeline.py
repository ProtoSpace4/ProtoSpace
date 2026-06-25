import numpy as np
import sys
sys.path.append('..')

def test_cdae_import():
    from pipeline.cdae import CDAE, add_noise, compute_snr
    assert CDAE is not None

def test_add_noise():
    from pipeline.cdae import add_noise
    flux = np.ones(200)
    noisy = add_noise(flux, noise_level=0.02)
    assert noisy.shape == flux.shape
    assert not np.allclose(flux, noisy)

def test_compute_snr():
    from pipeline.cdae import compute_snr
    raw = np.random.normal(1.0, 0.02, 500)
    clean = np.ones(500)
    result = compute_snr(raw, clean)
    assert 'snr_before' in result
    assert 'snr_after' in result
    assert 'improvement' in result

def test_preprocess_normalize():
    flux = np.array([100.0, 102.0, 98.0, 101.0])
    normalized = flux / np.nanmedian(flux)
    assert abs(np.median(normalized) - 1.0) < 0.01

print("All tests ready.")
