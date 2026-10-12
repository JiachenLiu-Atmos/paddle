"""Check that restart_regrid moves a restart onto a taller x1 domain correctly.

The source is a synthetic isothermal hydrostatic column, which has an exact
answer everywhere: rho and P fall as exp(-z/H) with a known H, so both the
interpolation inside the old domain and the extension above it can be checked
against the analytic profile rather than against themselves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from paddle import restart_regrid

NGHOST = 3
NVAR = 7  # rho, 3 velocities, energy, 2 mass fractions
H = 50.0e3  # scale height, m
RHO0 = 1.0e-2
CV_T = 2.0e5  # specific internal energy, J/kg
NX2 = 8


def config(x1max: float, nx1: int, nx2: int = NX2) -> dict:
    return {
        # Slot 5 is a vapour, slot 6 a condensate by Snapy's parenthesised name.
        "species": [{"name": "dry"}, {"name": "H2O"}, {"name": "H2O(l)"}],
        "boundary-condition": {
            "external": {"x1-inner": "reflecting", "x1-outer": "reflecting"}
        },
        "geometry": {
            "bounds": {
                "x1min": 0.0,
                "x1max": x1max,
                "x2min": 0.0,
                "x2max": 1.0e5,
            },
            "cells": {"nx1": nx1, "nx2": nx2, "nx3": 1, "nghost": NGHOST},
        },
    }


def analytic(z: np.ndarray) -> np.ndarray:
    return RHO0 * np.exp(-z / H)


def build_restart(grid, path: Path) -> None:
    """An isothermal column, uniform in x2, with ghosts mirrored about the walls."""
    z = grid.x1v()
    rho = analytic(z)
    u = np.zeros((NVAR, grid.nc3, grid.nc2, grid.nc1), dtype=np.float64)
    u[0] = rho
    u[1] = rho * 3.0  # a uniform x1 velocity
    u[4] = rho * CV_T
    u[5] = rho * 0.2  # mass fractions, constant with height
    u[6] = rho * 0.05
    w = np.zeros_like(u)
    w[0] = rho
    w[1] = 3.0
    w[4] = rho * CV_T * 0.4  # a pressure that falls like density
    w[5] = 0.2
    w[6] = 0.05
    # Snapy aliases fill_solid_* onto the hydro arrays when nothing is solid, and
    # the torch readers deduplicate by storage, so an aliased name can hide the
    # one it shares storage with. Reproduce that here rather than only in a run.
    hydro_u = torch.from_numpy(u)
    hydro_w = torch.from_numpy(w)
    restart_regrid.write_part(
        {
            "hydro_u": hydro_u,
            "hydro_w": hydro_w,
            "fill_solid_hydro_u": hydro_u,
            "fill_solid_hydro_w": hydro_w,
            "last_time": torch.tensor([1234.5], dtype=torch.float64),
            "last_cycle": torch.tensor([678], dtype=torch.int64),
            "file_number": torch.tensor([2, 3], dtype=torch.int64),
            "next_time": torch.tensor([10.0, 20.0], dtype=torch.float64),
        },
        str(path),
    )


OLD_CFG = config(600.0e3, 142)
NEW_CFG = config(760.0e3, 180)


@pytest.fixture(scope="module")
def source(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("regrid")
    (tmp / "old.yaml").write_text(yaml.safe_dump(OLD_CFG))
    (tmp / "new.yaml").write_text(yaml.safe_dump(NEW_CFG))
    build_restart(restart_regrid.Grid(OLD_CFG), tmp / "in.restart")
    return tmp


def run(source: Path, new_yaml: str, output: str, *extra: str) -> int:
    return restart_regrid.main(
        [
            "--old-config",
            str(source / "old.yaml"),
            "--new-config",
            str(source / new_yaml),
            "--restart",
            str(source / "in.restart"),
            "--output",
            str(source / output),
            *extra,
        ]
    )


@pytest.fixture(scope="module")
def taller(source: Path) -> dict[str, torch.Tensor]:
    assert run(source, "new.yaml", "out.restart") == 0
    return restart_regrid.read_part(str(source / "out.restart"))


def interior(tensors: dict[str, torch.Tensor], key: str) -> np.ndarray:
    """One column of active cells: x3 = 0, the first active x2, all active x1."""
    new = restart_regrid.Grid(NEW_CFG)
    arr = tensors[key].numpy()
    return arr[:, 0, NGHOST, NGHOST : NGHOST + new.nx1]


def heights() -> np.ndarray:
    new = restart_regrid.Grid(NEW_CFG)
    return new.x1v()[NGHOST : NGHOST + new.nx1]


def test_shape_follows_the_new_grid(taller) -> None:
    new = restart_regrid.Grid(NEW_CFG)
    assert tuple(taller["hydro_u"].shape) == (NVAR, new.nc3, new.nc2, new.nc1)


def test_aliased_names_all_survive_the_read(taller) -> None:
    assert {
        "hydro_u",
        "hydro_w",
        "fill_solid_hydro_u",
        "fill_solid_hydro_w",
    } <= set(taller)


def test_clock_is_carried_over(taller) -> None:
    assert taller["last_time"].item() == 1234.5
    assert taller["last_cycle"].item() == 678


def test_density_matches_the_analytic_column(taller) -> None:
    # The old domain ends at 600 km; above it the tool extrapolates, and an
    # isothermal column is exactly what it assumes, so one tolerance covers both.
    rho = interior(taller, "hydro_u")[0]
    error = np.abs(rho / analytic(heights()) - 1.0)
    assert error.max() < 2.0e-3


def test_extension_reaches_the_new_lid() -> None:
    assert heights()[-1] > 755.0e3


def test_per_mass_quantities_are_preserved(taller) -> None:
    u = interior(taller, "hydro_u")
    rho = u[0]
    assert np.abs(u[1] / rho - 3.0).max() < 1e-9
    assert np.abs(u[4] / rho - CV_T).max() / CV_T < 1e-9
    assert np.abs(u[5] / rho - 0.2).max() < 1e-12


def test_condensate_is_kept_inside_and_dropped_above(taller) -> None:
    old = restart_regrid.Grid(OLD_CFG)
    z_top = old.x1v()[NGHOST : NGHOST + old.nx1 - 2][-1]
    z = heights()
    u = interior(taller, "hydro_u")
    inside, above = z <= z_top, z > z_top
    assert np.abs(u[6][inside] / u[0][inside] - 0.05).max() < 1e-12
    assert above.any()
    assert np.abs(u[6][above]).max() == 0.0


def test_pressure_falls_like_density(taller) -> None:
    rho = interior(taller, "hydro_u")[0]
    pres = interior(taller, "hydro_w")[4]
    assert np.abs(pres / (rho * CV_T * 0.4) - 1.0).max() < 2.0e-3


def test_extension_stays_isothermal(taller) -> None:
    # P/rho is proportional to T/mu, so a constant one means the extension
    # above the old lid is isothermal rather than slowly drifting.
    w = interior(taller, "hydro_w")
    t_like = w[4] / w[0]
    assert np.abs(t_like / t_like[0] - 1.0).max() < 1.0e-9


@pytest.mark.parametrize("wall", ["bottom", "top"])
def test_x1_ghosts_mirror_the_wall(taller, wall) -> None:
    # A reflecting wall stores the mirror of the interior, with the wall-normal
    # velocity negated. Snapy does not refill these on restart, so the file
    # itself has to be right.
    u = taller["hydro_u"].numpy()[:, 0, NGHOST]
    if wall == "bottom":
        ghost, active = u[:, NGHOST - 1 :: -1], u[:, NGHOST : 2 * NGHOST]
    else:
        ghost, active = u[:, -NGHOST:], u[:, -NGHOST - 1 : -2 * NGHOST - 1 : -1]
    assert np.abs(ghost[0] - active[0]).max() < 1e-12
    assert np.abs(ghost[1] + active[1]).max() < 1e-12


def test_nx2_change_is_refused_without_stretch(source) -> None:
    (source / "wide.yaml").write_text(yaml.safe_dump(config(760.0e3, 180, NX2 + 4)))
    with pytest.raises(SystemExit, match="x2-mode stretch"):
        run(source, "wide.yaml", "wide.restart")


def test_nx2_change_is_accepted_with_stretch(source) -> None:
    (source / "wide.yaml").write_text(yaml.safe_dump(config(760.0e3, 180, NX2 + 4)))
    assert run(source, "wide.yaml", "stretched.restart", "--x2-mode", "stretch") == 0
    new = restart_regrid.Grid(NEW_CFG)
    stretched = restart_regrid.read_part(str(source / "stretched.restart"))
    assert tuple(stretched["hydro_u"].shape) == (
        NVAR,
        1,
        NX2 + 4 + 2 * NGHOST,
        new.nc1,
    )


# --- what the review of PR #47 asked to be guarded -------------------------


def write_variant(path: Path, grid, mutate) -> None:
    """The module fixture's column, with one thing changed by `mutate`."""
    z = grid.x1v()
    rho = analytic(z)
    u = np.zeros((NVAR, grid.nc3, grid.nc2, grid.nc1), dtype=np.float64)
    u[0] = rho
    u[1] = rho * 3.0
    u[4] = rho * CV_T
    u[5] = rho * 0.2
    u[6] = rho * 0.05
    w = np.zeros_like(u)
    w[0] = rho
    w[1] = 3.0
    w[4] = rho * CV_T * 0.4
    w[5] = 0.2
    w[6] = 0.05
    tensors = {
        "last_time": torch.tensor([1234.5], dtype=torch.float64),
        "last_cycle": torch.tensor([678], dtype=torch.int64),
    }
    mutate(z, u, w, tensors)
    tensors.setdefault("hydro_u", torch.from_numpy(u))
    tensors.setdefault("hydro_w", torch.from_numpy(w))
    restart_regrid.write_part(tensors, str(path))


def case(tmp_path: Path, mutate, old=OLD_CFG, new=NEW_CFG) -> Path:
    (tmp_path / "old.yaml").write_text(yaml.safe_dump(old))
    (tmp_path / "new.yaml").write_text(yaml.safe_dump(new))
    write_variant(tmp_path / "in.restart", restart_regrid.Grid(old), mutate)
    return tmp_path


def test_tracer_bearing_restart_is_refused(tmp_path) -> None:
    """scalar_s is laid out on x1 but is not regridded, so it must not pass."""

    def add_scalar(z, u, w, tensors):
        grid = restart_regrid.Grid(OLD_CFG)
        tensors["scalar_s"] = torch.zeros(
            (2, grid.nc3, grid.nc2, grid.nc1), dtype=torch.float64
        )

    src = case(tmp_path, add_scalar)
    with pytest.raises(SystemExit) as err:
        run(src, "new.yaml", "out.restart")
    assert "scalar_s" in str(err.value)
    assert not (src / "out.restart").exists()


def test_energy_stays_consistent_with_the_velocity(tmp_path) -> None:
    """E/rho - |v|^2/2 must still be the internal energy after the remap.

    Carrying E/rho across and multiplying by the new density does not survive a
    velocity that varies with height, because interpolation does not commute
    with squaring it. The column here has a constant internal energy, so the
    implied value is exactly CV_T wherever the answer is consistent.
    """

    def shear(z, u, w, tensors):
        vel = 3.0 + 1.0e-4 * z  # 0 -> 60 m/s over the column
        u[1] = u[0] * vel
        u[4] = u[0] * (CV_T + 0.5 * vel**2)
        w[1] = vel

    src = case(tmp_path, shear)
    assert run(src, "new.yaml", "out.restart") == 0
    out = restart_regrid.read_part(str(src / "out.restart"))
    col = interior(out, "hydro_u")
    implied = col[4] / col[0] - 0.5 * (col[1] / col[0]) ** 2
    assert np.allclose(implied, CV_T, rtol=1e-10)


def test_periodic_ghosts_wrap_the_active_cells(tmp_path) -> None:
    periodic = {
        **config(600.0e3, 142),
        "boundary-condition": {
            "external": {"x1-inner": "periodic", "x1-outer": "periodic"}
        },
    }
    new = {
        **config(760.0e3, 180),
        "boundary-condition": {
            "external": {"x1-inner": "periodic", "x1-outer": "periodic"}
        },
    }
    src = case(tmp_path, lambda *a: None, old=periodic, new=new)
    assert run(src, "new.yaml", "out.restart") == 0
    arr = restart_regrid.read_part(str(src / "out.restart"))["hydro_u"].numpy()
    nx1 = restart_regrid.Grid(new).nx1
    # the lower ghosts hold the topmost active cells, and the reverse
    assert np.array_equal(arr[..., :NGHOST], arr[..., nx1 : nx1 + NGHOST])
    assert np.array_equal(arr[..., -NGHOST:], arr[..., NGHOST : 2 * NGHOST])


def test_one_sided_periodic_is_refused(tmp_path) -> None:
    half = {
        **config(600.0e3, 142),
        "boundary-condition": {
            "external": {"x1-inner": "periodic", "x1-outer": "reflecting"}
        },
    }
    src = case(tmp_path, lambda *a: None, old=half, new=half)
    with pytest.raises(SystemExit, match="both ends"):
        run(src, "new.yaml", "out.restart")


def test_unsupported_boundary_is_refused(tmp_path) -> None:
    odd = {
        **config(600.0e3, 142),
        "boundary-condition": {
            "external": {"x1-inner": "shear-periodic", "x1-outer": "reflecting"}
        },
    }
    src = case(tmp_path, lambda *a: None, old=odd, new=odd)
    with pytest.raises(SystemExit, match="does not rebuild"):
        run(src, "new.yaml", "out.restart")


def test_ghost_width_change_is_refused(tmp_path) -> None:
    wider = config(760.0e3, 180)
    wider["geometry"]["cells"]["nghost"] = 2
    src = case(tmp_path, lambda *a: None, new=wider)
    with pytest.raises(SystemExit, match="nghost must be unchanged"):
        run(src, "new.yaml", "out.restart")


def test_non_finite_input_is_refused(tmp_path) -> None:
    def poison(z, u, w, tensors):
        u[0, 0, NGHOST, NGHOST + 10] = np.nan

    src = case(tmp_path, poison)
    with pytest.raises(ValueError, match="non-finite"):
        run(src, "new.yaml", "out.restart")
    assert not (src / "out.restart").exists()


def test_existing_output_is_kept_unless_forced(tmp_path) -> None:
    src = case(tmp_path, lambda *a: None)
    (src / "out.restart").write_bytes(b"do not lose me")
    with pytest.raises(SystemExit, match="already exists"):
        run(src, "new.yaml", "out.restart")
    assert (src / "out.restart").read_bytes() == b"do not lose me"
    assert run(src, "new.yaml", "out.restart", "--force") == 0
    assert (src / "out.restart").read_bytes() != b"do not lose me"


def test_input_and_output_must_differ(tmp_path) -> None:
    src = case(tmp_path, lambda *a: None)
    before = (src / "in.restart").read_bytes()
    with pytest.raises(SystemExit, match="same file"):
        run(src, "new.yaml", "in.restart")
    assert (src / "in.restart").read_bytes() == before


def test_non_isothermal_edge_is_refused_then_allowed(tmp_path) -> None:
    """A column with a real lapse rate cannot get an isothermal extension."""

    def lapse(z, u, w, tensors):
        # P/rho falls with height instead of staying flat
        w[4] = u[0] * CV_T * 0.4 * (1.0 - 0.6 * z / 600.0e3)

    src = case(tmp_path, lapse)
    with pytest.raises(ValueError, match="hydrostatic balance"):
        run(src, "new.yaml", "out.restart")
    assert run(src, "new.yaml", "out.restart", "--allow-nonisothermal") == 0


# --- the second review round ----------------------------------------------


def test_energy_survives_the_horizontal_resample(tmp_path) -> None:
    """Rebuilding the conserved state before the x2 map undoes the x1 fix.

    resample_x2_periodic interpolates every slot on its own, so a momentum and
    an energy that agreed after the vertical map stop agreeing after the
    horizontal one unless both ride across in per-mass form.
    """

    def shear_in_x2(z, u, w, tensors):
        grid = restart_regrid.Grid(OLD_CFG)
        # velocity varies along x2 as well as x1, so the horizontal resample
        # has something to get wrong
        j = np.arange(grid.nc2).reshape(1, -1, 1)
        vel = 3.0 + 1.0e-4 * z + 2.0 * j
        u[1] = u[0] * vel
        u[4] = u[0] * (CV_T + 0.5 * vel**2)
        w[1] = vel

    wide = config(760.0e3, 180, nx2=12)
    src = case(tmp_path, shear_in_x2, new=wide)
    assert run(src, "new.yaml", "out.restart", "--x2-mode", "stretch") == 0
    arr = restart_regrid.read_part(str(src / "out.restart"))["hydro_u"].numpy()
    grid = restart_regrid.Grid(wide)
    act = arr[:, 0, NGHOST : NGHOST + grid.nx2, NGHOST : NGHOST + grid.nx1]
    implied = act[4] / act[0] - 0.5 * (act[1] / act[0]) ** 2
    assert np.allclose(implied, CV_T, rtol=1e-10)


def test_temperature_guard_ignores_horizontal_ghosts(tmp_path) -> None:
    """The lapse-rate guard describes the column, not the x2 ghost zones."""

    def poison_x2_ghosts(z, u, w, tensors):
        w[4, :, :NGHOST, :] *= 5.0  # nothing physical lives here
        w[4, :, -NGHOST:, :] *= 5.0

    src = case(tmp_path, poison_x2_ghosts)
    assert run(src, "new.yaml", "out.restart") == 0


def test_guard_window_matches_the_slope_fit(tmp_path, capsys) -> None:
    """The guard reports the window fit_top_slope will use, not --edge-fit raw.

    fit_top_slope floors the window at two cells, because a slope needs two
    points. A guard that took --edge-fit literally would describe a different
    set of cells from the one the extension is built out of, and at
    --edge-fit 1 it would be reducing over a single value.
    """
    src = case(tmp_path, lambda *a: None)
    assert run(src, "new.yaml", "out.restart", "--edge-fit", "1") == 0
    reported = capsys.readouterr().out
    assert "over the top 2 fitted cells" in reported
    assert restart_regrid.fit_top_slope(np.arange(5.0), np.arange(5.0), 1) == pytest.approx(1.0)


def test_outflow_ghosts_hold_the_edge_value(tmp_path) -> None:
    out_cfg = {
        **config(600.0e3, 142),
        "boundary-condition": {
            "external": {"x1-inner": "outflow", "x1-outer": "outflow"}
        },
    }
    new_cfg = {
        **config(760.0e3, 180),
        "boundary-condition": {
            "external": {"x1-inner": "outflow", "x1-outer": "outflow"}
        },
    }
    src = case(tmp_path, lambda *a: None, old=out_cfg, new=new_cfg)
    assert run(src, "new.yaml", "out.restart") == 0
    arr = restart_regrid.read_part(str(src / "out.restart"))["hydro_u"].numpy()
    nx1 = restart_regrid.Grid(new_cfg).nx1
    # zero gradient: every ghost equals the active cell it sits against
    for g in range(NGHOST):
        assert np.array_equal(arr[..., g], arr[..., NGHOST])
        assert np.array_equal(arr[..., -1 - g], arr[..., NGHOST + nx1 - 1])


def test_overflow_in_the_output_type_is_refused() -> None:
    """A value finite in float64 and infinite in float32 must not be written."""
    old = restart_regrid.Grid(OLD_CFG)
    new = restart_regrid.Grid(NEW_CFG)
    roles = restart_regrid.variable_roles(NVAR, "default")
    z = old.x1v()
    # density climbing steeply with height: the extension continues the fitted
    # log slope and leaves float32 behind without leaving float64
    arr = np.zeros((NVAR, old.nc3, old.nc2, old.nc1), dtype=np.float64)
    arr[0] = 1.0e30 * np.exp(z / 8.0e3)
    arr[4] = arr[0] * CV_T

    kwargs = dict(roles=roles, conserved=True, skip=2, nfit=12)
    full = restart_regrid.regrid_state(arr, old, new, **kwargs, dtype=np.float64)
    assert np.all(np.isfinite(full))
    with pytest.raises(ValueError, match="float32"):
        restart_regrid.regrid_state(arr, old, new, **kwargs, dtype=np.float32)


def test_guard_allows_a_horizontal_temperature_gradient(tmp_path) -> None:
    """Each column isothermal, the columns at different temperatures.

    map_x1 fits and extends one column at a time, so a horizontal gradient is
    something the extension handles. A guard that reduced across columns would
    read it as a lapse rate and refuse a state the tool maps correctly.
    """

    def warm_one_side(z, u, w, tensors):
        grid = restart_regrid.Grid(OLD_CFG)
        # column j sits at its own temperature, flat in height
        j = np.arange(grid.nc2).reshape(1, -1, 1)
        w[4] = u[0] * CV_T * 0.4 * (1.0 + 0.5 * j / grid.nc2)

    src = case(tmp_path, warm_one_side)
    assert run(src, "new.yaml", "out.restart") == 0


def test_guard_still_catches_a_lapse_in_one_column(tmp_path) -> None:
    """Per column does not mean per plane: one bad column is enough."""

    def one_column_with_lapse(z, u, w, tensors):
        w[4] = u[0] * CV_T * 0.4
        w[4, :, NGHOST + 2, :] = u[0, :, NGHOST + 2, :] * (
            CV_T * 0.4 * (1.0 - 0.6 * z / 600.0e3)
        )

    src = case(tmp_path, one_column_with_lapse)
    with pytest.raises(ValueError, match="one column spans"):
        run(src, "new.yaml", "out.restart")
