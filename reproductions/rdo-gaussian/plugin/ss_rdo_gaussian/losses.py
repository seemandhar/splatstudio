"""RDO-Gaussian loss terms (paper Eq. 13; official train.py

    loss = render_loss + vq_loss + rate_loss + sh_mask_lambda * sh_mask_loss
           + gs_mask_lambda * gs_mask_loss).

The renderer computes the terms (`outputs["rdo"]`, official normalisation); each loss here
returns `weight * term`, or an attached zero before the RD stage starts (step <= 15000).
"""



def _make(term, default_weight):
    def factory(spec):
        unknown = set(spec.config) - {"weight"}
        if unknown:
            raise ValueError(f"Unknown {spec.name} options: {sorted(unknown)}")
        weight = float(spec.config.get("weight", default_weight))

        def loss(params, outputs, target):
            rdo = outputs.get("rdo")
            if rdo is None:
                return outputs["rgb"].sum() * 0.0
            return weight * rdo[term]

        return loss

    return factory


create_rate = _make("rate", 1.0)  # L_rate (Eq. 11)
create_vq = _make("vq", 1.0)  # L_VQ (Eq. 11)
create_gs_prune = _make("gs_prune", 0.0005)  # lambda_GSprune L_GSprune (Eq. 5), highest rate
create_sh_prune = _make("sh_prune", 0.005)  # lambda_SHprune L_SHprune (Eq. 8), highest rate
