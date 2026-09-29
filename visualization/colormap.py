"""NumPy translation of gptoolbox's requested OKLab striped colormap.

Sources: gptoolbox/imageprocessing/{okloop,oklab2lin,lin2oklab}.m and
mesh/isolines_stripe_map.m. See third_party/gptoolbox-LICENSE-MIT.txt.
Colors are evaluated in sRGB; Blender receives their linear-light values.
"""

import numpy as np


def oklab_to_linear(ok):
    lightness, a, b = np.asarray(ok).T
    lms = (
        np.column_stack(
            (
                lightness + 0.3963377774 * a + 0.2158037573 * b,
                lightness - 0.1055613458 * a - 0.0638541728 * b,
                lightness - 0.0894841775 * a - 1.2914855480 * b,
            )
        )
        ** 3
    )
    return lms @ np.array(
        [
            [4.0767245293, -1.2681437731, -0.0041119885],
            [-3.3072168827, 2.6093323231, -0.7034763098],
            [0.2307590544, -0.3411344290, 1.7068625689],
        ]
    )


def linear_to_oklab(rgb):
    lms = np.cbrt(
        np.asarray(rgb)
        @ np.array(
            [
                [0.4121656120, 0.2118591070, 0.0883097947],
                [0.5362752080, 0.6807189584, 0.2818474174],
                [0.0514575653, 0.1074065790, 0.6302613616],
            ]
        )
    )
    return lms @ np.array(
        [
            [0.2104542553, 1.9779984951, 0.0259040371],
            [0.7936177850, -2.4285922050, 0.7827717662],
            [-0.0040720468, 0.4505937099, -0.8086757660],
        ]
    )


def linear_to_srgb(x):
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.maximum(x, 0) ** (1 / 2.4) - 0.055)


def striped_okloop():
    theta = -np.pi / 2 + np.arange(26) / 26 * (-4 * np.pi / 3)
    ok = np.column_stack(
        (
            np.full(26, 0.75010101010101016),
            0.12755316371916220 * np.cos(theta),
            0.12755316371916220 * np.sin(theta),
        )
    )
    # Match isolines_stripe_map's RGB->OKLab round trip before alternating L.
    ok = linear_to_oklab(oklab_to_linear(ok))
    ok[1::2, 0] *= 0.9
    linear = np.clip(oklab_to_linear(ok), 0, 1)
    return linear, np.clip(linear_to_srgb(linear), 0, 1)
