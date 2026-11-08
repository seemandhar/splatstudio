from ss_mcmc_3dgs import method, probes


def test_cap_lookup():
    probes.cap_lookup(None)


def test_official_init():
    probes.official_init(None)


def test_regularizers():
    probes.regularizers(None)


def test_method_builds():
    assert method().name == "mcmc-3dgs"
