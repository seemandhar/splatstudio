from ss_gsurfels import method, method_nogeo, probes


def test_consistency_formula():
    probes.consistency_formula(None, "cpu")


def test_normal_prior_formula():
    probes.normal_prior_formula(None, "cpu")


def test_opacity_formula():
    probes.opacity_formula(None, "cpu")


def test_schedule_recipe():
    probes.schedule_recipe(None, "cpu")


def test_density_recipe_cpu():
    probes.density_recipe(None, "cpu")


def test_specs():
    full = {c.name for c in method().components}
    assert {"gsurfels-consistency", "gsurfels-normal-prior", "gsurfels-opacity"} <= full
    nogeo = {c.name for c in method_nogeo().components}
    assert not {"gsurfels-consistency", "gsurfels-normal-prior", "gsurfels-opacity"} & nogeo
    assert method().representation == "surfel2d"
