"""
modules/plant.py — PMSM plant model in d-q frame.

Supports SPMSM (Ld = Lq) and IPMSM (Ld < Lq).
Provides continuous state-space, discretization, MTPA, and field weakening.

Voltage equations (motor convention, d-q frame):
    vd = Rs*id + Ld*d(id)/dt - omega_e*Lq*iq
    vq = Rs*iq + Lq*d(iq)/dt + omega_e*(Ld*id + psi_f)

Torque and mechanical equations:
    Te = (3/2)*p*(psi_f*iq + (Ld - Lq)*id*iq)
    J_total*d(omega_m)/dt = Te - TL - B_total*omega_m
    omega_e = p * omega_m
"""

import math

import numpy as np
from scipy.signal import cont2discrete

# ---------------------------------------------------------------------------
# Named constants — thresholds used in warnings / checks
# ---------------------------------------------------------------------------
_SPMSM_SALIENCY_TOL = 0.05          # allow up to 5% Ld/Lq difference for SPMSM
_FW_VOLTAGE_MARGIN = 0.95           # use 95% of Vdc/sqrt(3) as voltage limit


class PMSMPlant:
    """
    Continuous and discrete PMSM plant in d-q frame.

    Parameters
    ----------
    motor_cfg : dict
        Validated motor config from utils.config.load_config().
    load_cfg : dict
        Validated load config from utils.config.load_config().
    """

    def __init__(self, motor_cfg: dict, load_cfg: dict) -> None:
        self.motor_cfg = motor_cfg
        self.load_cfg = load_cfg
        self.motor_type: str = motor_cfg["motor_type"]   # 'SPMSM' or 'IPMSM'

        # Electrical parameters (SI)
        elec = motor_cfg["electrical"]
        self.Rs: float = elec["Rs_ohm"]
        self.Ld: float = elec["Ld_H"]
        self.Lq: float = elec["Lq_H"]
        self.psi_f: float = elec["psi_f_Wb"]
        self.p: int = elec["pole_pairs"]

        # Mechanical parameters (SI)
        mech = motor_cfg["mechanical"]
        self.J_motor: float = mech["J_kgm2"]
        self.B_motor: float = mech["B_Nms_rad"]

        # Load parameters (SI)
        self.J_load: float = load_cfg["J_load_kgm2"]
        self.B_load: float = load_cfg["B_load_Nms_rad"]
        self.load_type: str = load_cfg["load_type"]
        self.k_fan: float = load_cfg["k_fan"]
        self.TL_const: float = load_cfg["TL_Nm"]

        # Combined mechanical parameters
        self.J_total: float = self.J_motor + self.J_load
        self.B_total: float = self.B_motor + self.B_load

        # SPMSM: enforce Ld = Lq (already averaged in config loader, double-check)
        if self.motor_type == "SPMSM":
            self.Lq = self.Ld

    # ------------------------------------------------------------------
    # State-space (current plant)
    # ------------------------------------------------------------------

    def build_state_space(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build continuous-time state-space matrices for the 2-axis current plant.

        State vector:  x = [id, iq]
        Input vector:  u = [vd, vq]
        Output vector: y = [id, iq]

        Cross-coupling (omega_e * L * i) and back-EMF are treated as
        disturbances and are NOT included in A/B — they are compensated
        by the decoupling feedforward terms in the controller.

        Per-axis first-order plants:
            d-axis: G_d(s) = (1/Rs) / (tau_d*s + 1),  tau_d = Ld/Rs
            q-axis: G_q(s) = (1/Rs) / (tau_q*s + 1),  tau_q = Lq/Rs

        State equations (decoupled approximation):
            d(id)/dt = -Rs/Ld * id + 1/Ld * vd
            d(iq)/dt = -Rs/Lq * iq + 1/Lq * vq

        Returns
        -------
        A_c : ndarray, shape (2, 2)
        B_c : ndarray, shape (2, 2)
        C_c : ndarray, shape (2, 2)  — identity (both states are outputs)
        """
        A_c = np.array([
            [-self.Rs / self.Ld,  0.0             ],
            [ 0.0,                -self.Rs / self.Lq],
        ])
        B_c = np.array([
            [1.0 / self.Ld,  0.0           ],
            [0.0,             1.0 / self.Lq],
        ])
        C_c = np.eye(2)
        return A_c, B_c, C_c

    def get_current_plant_tf(self, axis: str = 'd') -> tuple[np.ndarray, np.ndarray]:
        """
        Return (num, den) of the per-axis current plant transfer function.

        G_d(s) = (1/Rs) / (tau_d*s + 1)
        G_q(s) = (1/Rs) / (tau_q*s + 1)

        Parameters
        ----------
        axis : 'd' or 'q'

        Returns
        -------
        num : ndarray   [1/Rs]
        den : ndarray   [tau, 1]
        """
        if axis == 'd':
            tau = self.Ld / self.Rs
        elif axis == 'q':
            tau = self.Lq / self.Rs
        else:
            raise ValueError(f"axis must be 'd' or 'q', got '{axis}'")

        num = np.array([1.0 / self.Rs])
        den = np.array([tau, 1.0])
        return num, den

    def get_speed_plant_tf(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (num, den) of the mechanical speed plant transfer function.

        Assumes current loop is closed and approximated as unity gain
        (valid for omega_speed << BW_current). Torque constant:
            Kt_eff = (3/2) * p * psi_f   [N·m/A, for iq with id=0]

        G_speed(s) = Kt_eff / (J_total * s + B_total)
                   = (Kt_eff / J_total) / (s + B_total/J_total)

        For loop design (B_total often negligible vs J_total * s):
            G_speed(s) ≈ Kt_eff / (J_total * s)   [integrator approximation]

        Returns
        -------
        num : ndarray   [Kt_eff]
        den : ndarray   [J_total, B_total]
        """
        Kt_eff = 1.5 * self.p * self.psi_f
        num = np.array([Kt_eff])
        den = np.array([self.J_total, self.B_total])
        return num, den

    # ------------------------------------------------------------------
    # Discretization
    # ------------------------------------------------------------------

    def discretize(
        self, Ts_s: float, method: str = "zoh"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Discretize the current plant state-space using ZOH or Tustin.

        Parameters
        ----------
        Ts_s : float
            Sampling period in seconds (e.g., 50e-6 for 20 kHz).
        method : str
            'zoh' (zero-order hold) or 'tustin' (bilinear).

        Returns
        -------
        A_d : ndarray, shape (2, 2)
        B_d : ndarray, shape (2, 2)
        """
        if Ts_s <= 0:
            raise ValueError(f"Ts_s must be > 0, got {Ts_s}")
        if method not in ("zoh", "tustin"):
            raise ValueError(f"method must be 'zoh' or 'tustin', got '{method}'")

        A_c, B_c, C_c = self.build_state_space()
        D_c = np.zeros((2, 2))

        result = cont2discrete((A_c, B_c, C_c, D_c), Ts_s, method=method)
        A_d, B_d = result[0], result[1]
        return A_d, B_d

    # ------------------------------------------------------------------
    # MTPA (IPMSM only)
    # ------------------------------------------------------------------

    def mtpa_angle(self, Is: float) -> float:
        """
        Compute MTPA current angle for IPMSM (returns 0.0 for SPMSM).

        Maximum Torque Per Ampere angle beta such that:
            id* = -Is * sin(beta)
            iq* =  Is * cos(beta)

        Formula (from differentiation of Te w.r.t. beta):
            id_MTPA = (psi_f - sqrt(psi_f^2 + 8*(Lq-Ld)^2*Is^2))
                      / (4*(Lq-Ld))

        Parameters
        ----------
        Is : float
            Current magnitude in A (>= 0).

        Returns
        -------
        beta : float
            MTPA angle in radians. 0.0 for SPMSM.
        """
        if self.motor_type == "SPMSM":
            return 0.0

        if Is <= 0:
            return 0.0

        dL = self.Lq - self.Ld   # > 0 for IPMSM
        id_mtpa = (
            self.psi_f - math.sqrt(self.psi_f**2 + 8.0 * dL**2 * Is**2)
        ) / (4.0 * dL)

        # beta = arcsin(-id_mtpa / Is)  — id_mtpa is negative
        beta = math.asin(-id_mtpa / Is)
        return beta

    def mtpa_currents(self, Is: float) -> tuple[float, float]:
        """
        Return (id*, iq*) at MTPA operating point for given current magnitude.

        Parameters
        ----------
        Is : float
            Current magnitude in A.

        Returns
        -------
        id_star : float   (0.0 for SPMSM, negative for IPMSM)
        iq_star : float   (= Is for SPMSM)
        """
        beta = self.mtpa_angle(Is)
        id_star = -Is * math.sin(beta)
        iq_star = Is * math.cos(beta)
        return id_star, iq_star

    # ------------------------------------------------------------------
    # Field weakening (IPMSM / SPMSM above base speed)
    # ------------------------------------------------------------------

    def field_weakening_id(
        self, omega_e: float, Vdc: float, Is_max: float
    ) -> float:
        """
        Compute id* required for field weakening above base speed.

        Operating constraints:
            Voltage limit:  Vd^2 + Vq^2 <= (Vdc / sqrt(3))^2   [space-vector limit]
            Current limit:  id^2 + iq^2 <= Is_max^2

        At the voltage limit boundary (steady-state, id=0 control until base speed):
            Vs_max = Vdc / sqrt(3) * _FW_VOLTAGE_MARGIN
            |Vs|^2 = (omega_e * Lq * iq)^2 + (omega_e * (Ld*id + psi_f))^2

        Solving for id* at current limit (iq* = sqrt(Is_max^2 - id*^2)):
            id* = -(psi_f/Ld) + sqrt((Vs_max/(omega_e*Ld))^2 - (Lq/Ld*iq*)^2)
            (iterative or simplified: assume iq* = Is_max for first estimate)

        Simplified single-step solution (valid near base speed):
            id_fw = -(psi_f - Vs_max/omega_e) / Ld
        Clamp: id_fw in [-Is_max, 0].

        Parameters
        ----------
        omega_e : float
            Electrical angular speed in rad/s.
        Vdc : float
            DC link voltage in V.
        Is_max : float
            Maximum current magnitude in A.

        Returns
        -------
        id_star : float
            Field-weakening d-axis current command in A (<= 0).
        """
        if omega_e <= 0:
            return 0.0

        Vs_max = (Vdc / math.sqrt(3.0)) * _FW_VOLTAGE_MARGIN

        # Simplified: ignore cross-coupling (good first estimate)
        id_fw = -(self.psi_f - Vs_max / omega_e) / self.Ld

        # Clamp to [-Is_max, 0]
        id_fw = max(-Is_max, min(0.0, id_fw))
        return id_fw

    # ------------------------------------------------------------------
    # Torque and load
    # ------------------------------------------------------------------

    def electromagnetic_torque(self, id: float, iq: float) -> float:
        """
        Compute electromagnetic torque.

        Te = (3/2) * p * (psi_f * iq + (Ld - Lq) * id * iq)

        For SPMSM: Ld = Lq → reluctance term = 0 → Te = (3/2)*p*psi_f*iq.

        Parameters
        ----------
        id : float   d-axis current in A
        iq : float   q-axis current in A

        Returns
        -------
        Te : float   Electromagnetic torque in N·m
        """
        return 1.5 * self.p * (self.psi_f * iq + (self.Ld - self.Lq) * id * iq)

    def load_torque(self, omega_m: float) -> float:
        """
        Compute load torque for the configured load type.

        fan:             TL = k_fan * omega_m^2
        constant_torque: TL = TL_const
        position_servo:  TL = 0

        Parameters
        ----------
        omega_m : float
            Mechanical angular speed in rad/s.

        Returns
        -------
        TL : float   Load torque in N·m
        """
        if self.load_type == "fan":
            return self.k_fan * omega_m**2
        elif self.load_type == "constant_torque":
            return self.TL_const
        else:   # position_servo
            return 0.0

    def mechanical_derivative(
        self, omega_m: float, id: float, iq: float
    ) -> float:
        """
        Compute d(omega_m)/dt from the mechanical equation of motion.

        J_total * d(omega_m)/dt = Te - TL(omega_m) - B_total * omega_m

        Parameters
        ----------
        omega_m : float   Mechanical speed in rad/s
        id : float        d-axis current in A
        iq : float        q-axis current in A

        Returns
        -------
        domega_dt : float   Angular acceleration in rad/s^2
        """
        Te = self.electromagnetic_torque(id, iq)
        TL = self.load_torque(omega_m)
        return (Te - TL - self.B_total * omega_m) / self.J_total

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a concise parameter summary to stdout."""
        print(f"PMSMPlant: {self.motor_cfg['name']}  [{self.motor_type}]")
        print(f"  Rs={self.Rs} ohm  Ld={self.Ld*1e3:.3f} mH  Lq={self.Lq*1e3:.3f} mH")
        print(f"  psi_f={self.psi_f*1e3:.2f} mWb  p={self.p}")
        print(f"  tau_d={self.Ld/self.Rs*1e3:.2f} ms  tau_q={self.Lq/self.Rs*1e3:.2f} ms")
        print(f"  J_total={self.J_total:.4e} kg*m^2  B_total={self.B_total:.4e} N*m*s/rad")
        print(f"  Load: {self.load_cfg['name']}  [{self.load_type}]")
        Kt_eff = 1.5 * self.p * self.psi_f
        print(f"  Kt_eff (3/2*p*psi_f) = {Kt_eff:.4f} N*m/A")
