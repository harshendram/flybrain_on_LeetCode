// On-screen stick (bottom-left) plus climb buttons. Pointer events stay on the stick so a drag
// there does not orbit the camera. Any touch counts as "the user is flying".
import type { FlightInput } from "./pilot";

export class Joystick {
  thrust = 0;
  yaw = 0;
  climb = 0;
  touched = false;
  readonly el: HTMLElement;

  constructor(parent: HTMLElement) {
    const root = document.createElement("div");
    root.className = "joystick";
    root.innerHTML =
      `<div class="stick" id="stick"><div class="knob"></div></div>` +
      `<div class="climb-pad"><button type="button" data-climb="1" aria-label="Climb">▲</button>` +
      `<button type="button" data-climb="-1" aria-label="Descend">▼</button></div>`;
    parent.appendChild(root);
    this.el = root;
    const stick = root.querySelector<HTMLElement>(".stick")!;
    const knob = root.querySelector<HTMLElement>(".knob")!;
    const set = (clientX: number, clientY: number) => {
      const box = stick.getBoundingClientRect();
      const x = Math.max(-1, Math.min(1, (clientX - (box.left + box.width / 2)) / (box.width / 2)));
      const y = Math.max(-1, Math.min(1, (clientY - (box.top + box.height / 2)) / (box.height / 2)));
      this.yaw = x;
      this.thrust = -y;
      knob.style.transform = `translate(${x * 28}px, ${y * 28}px)`;
      this.touched = true;
    };
    const clear = () => {
      this.yaw = 0;
      this.thrust = 0;
      knob.style.transform = "";
    };
    const stop = (e: Event) => e.stopPropagation();
    stick.addEventListener("pointerdown", (e) => {
      stick.setPointerCapture(e.pointerId);
      set(e.clientX, e.clientY);
      stop(e);
    });
    stick.addEventListener("pointermove", (e) => {
      if (stick.hasPointerCapture(e.pointerId)) set(e.clientX, e.clientY);
    });
    stick.addEventListener("pointerup", clear);
    stick.addEventListener("pointercancel", clear);
    root.addEventListener("pointerdown", stop);

    for (const btn of root.querySelectorAll<HTMLButtonElement>("[data-climb]")) {
      const v = Number(btn.dataset.climb);
      const on = (e: Event) => {
        this.climb = v;
        this.touched = true;
        e.preventDefault();
        stop(e);
      };
      const off = () => {
        if (this.climb === v) this.climb = 0;
      };
      btn.addEventListener("pointerdown", on);
      btn.addEventListener("pointerup", off);
      btn.addEventListener("pointerleave", off);
      btn.addEventListener("pointercancel", off);
    }
  }

  input(): FlightInput {
    return { thrust: this.thrust, yaw: this.yaw, climb: this.climb };
  }

  add(keys: FlightInput): FlightInput {
    return {
      thrust: Math.max(-1, Math.min(1, keys.thrust + this.thrust)),
      yaw: Math.max(-1, Math.min(1, keys.yaw + this.yaw)),
      climb: Math.max(-1, Math.min(1, keys.climb + this.climb)),
    };
  }
}
