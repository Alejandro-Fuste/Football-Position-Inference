#!/usr/bin/env python3
"""Interactive field-correspondence calibration GUI for the field-context PoC.

The tool opens a single formation-frame image, lets the user click visible field
intersections, prompts for the corresponding metric field coordinate in yards, and
writes JSON consumed by ``scripts/field_context_poc.py``.

Important design choices:
- Saved image coordinates always refer to the ORIGINAL image resolution, even when the
  GUI scales the image to fit the screen.
- This utility does not use player-position ground truth.
- This utility does not modify the production position-inference pipeline.
- At least four non-collinear correspondences are required for a homography; 6-8 well
  distributed yard-line/hash-mark intersections are recommended.

Example:
    python scripts/field_context_calibrator.py \
      --image data/JetSweep_2_frame0.jpg \
      --output scripts/JetSweep_2_field_correspondences.json \
      --video-id JetSweep_2 \
      --field-type NFL

Controls:
- Left-click: add a correspondence and enter ``field_x, field_y`` in yards.
- Undo button or ``u``: remove the most recent point.
- Save button or ``s``: save the current JSON.
- Finish button or ``q``: save (if >=4 points) and close.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

try:
    import tkinter as tk
    from tkinter import messagebox, simpledialog
except ImportError as exc:  # pragma: no cover - platform dependency
    raise SystemExit(
        "Tkinter is required for the calibration GUI. On macOS, the Python.org Python "
        "installer normally includes Tkinter."
    ) from exc

try:
    from PIL import Image, ImageTk
except ImportError as exc:  # pragma: no cover - optional PoC dependency
    raise SystemExit(
        'Pillow is required for this PoC GUI. Install the optional tools with:\n'
        '  pip install -e ".[field-poc]"\n'
        "or install Pillow directly with:\n"
        "  pip install pillow"
    ) from exc


class CalibrationGUI:
    def __init__(
        self,
        root: tk.Tk,
        image_path: Path,
        output_path: Path,
        video_id: str,
        field_type: str,
    ) -> None:
        self.root = root
        self.image_path = image_path
        self.output_path = output_path
        self.video_id = video_id
        self.field_type = field_type
        self.points: List[Dict[str, object]] = []

        self.original = Image.open(image_path).convert("RGB")
        self.original_w, self.original_h = self.original.size

        self.root.title(f"Field Context Calibrator — {video_id}")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        screen_w = max(900, self.root.winfo_screenwidth())
        screen_h = max(700, self.root.winfo_screenheight())
        max_w = int(screen_w * 0.78)
        max_h = int(screen_h * 0.78)
        self.scale = min(max_w / self.original_w, max_h / self.original_h, 1.0)
        self.display_w = max(1, int(round(self.original_w * self.scale)))
        self.display_h = max(1, int(round(self.original_h * self.scale)))

        if self.scale < 1.0:
            display_image = self.original.resize(
                (self.display_w, self.display_h),
                Image.Resampling.LANCZOS,
            )
        else:
            display_image = self.original

        self.photo = ImageTk.PhotoImage(display_image)

        top = tk.Frame(root)
        top.pack(fill=tk.X, padx=8, pady=(8, 4))

        instructions = (
            "Click a visible yard-line/hash-mark intersection. Then enter its field coordinate "
            "as: field_x, field_y (yards). Aim for 6–8 points spread across the field."
        )
        tk.Label(top, text=instructions, anchor="w", justify=tk.LEFT).pack(fill=tk.X)
        tk.Label(
            top,
            text=(
                f"Image: {image_path.name}  |  original: {self.original_w}×{self.original_h}  "
                f"|  display scale: {self.scale:.3f}"
            ),
            anchor="w",
        ).pack(fill=tk.X, pady=(2, 0))

        body = tk.Frame(root)
        body.pack(fill=tk.BOTH, expand=True, padx=8)

        self.canvas = tk.Canvas(
            body,
            width=self.display_w,
            height=self.display_h,
            highlightthickness=1,
            highlightbackground="gray",
            cursor="crosshair",
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo, tags=("base_image",))
        self.canvas.bind("<Button-1>", self.on_click)

        side = tk.Frame(body, width=310)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        side.pack_propagate(False)

        tk.Label(side, text="Selected correspondences", font=("TkDefaultFont", 12, "bold")).pack(
            anchor="w", pady=(0, 4)
        )
        self.listbox = tk.Listbox(side, width=46, height=24)
        self.listbox.pack(fill=tk.BOTH, expand=True)

        button_row = tk.Frame(side)
        button_row.pack(fill=tk.X, pady=(8, 0))
        tk.Button(button_row, text="Undo (U)", command=self.undo).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(button_row, text="Save (S)", command=self.save).pack(side=tk.LEFT, padx=4)
        tk.Button(button_row, text="Finish (Q)", command=self.finish).pack(side=tk.LEFT, padx=4)

        self.status_var = tk.StringVar(value="0 points selected")
        tk.Label(side, textvariable=self.status_var, anchor="w", justify=tk.LEFT).pack(
            fill=tk.X, pady=(8, 0)
        )

        self.root.bind("u", lambda _e: self.undo())
        self.root.bind("U", lambda _e: self.undo())
        self.root.bind("s", lambda _e: self.save())
        self.root.bind("S", lambda _e: self.save())
        self.root.bind("q", lambda _e: self.finish())
        self.root.bind("Q", lambda _e: self.finish())

        self._load_existing_if_present()
        self.redraw()

    def _load_existing_if_present(self) -> None:
        if not self.output_path.exists():
            return
        try:
            payload = json.loads(self.output_path.read_text())
            raw_points = payload.get("points", [])
            valid_points: List[Dict[str, object]] = []
            for point in raw_points:
                image = point.get("image")
                field = point.get("field")
                if (
                    isinstance(image, list)
                    and len(image) == 2
                    and isinstance(field, list)
                    and len(field) == 2
                ):
                    valid_points.append(
                        {
                            "image": [float(image[0]), float(image[1])],
                            "field": [float(field[0]), float(field[1])],
                        }
                    )
            if valid_points:
                self.points = valid_points
                messagebox.showinfo(
                    "Existing calibration loaded",
                    f"Loaded {len(valid_points)} existing point(s) from:\n{self.output_path}",
                )
        except Exception as exc:
            messagebox.showwarning(
                "Could not load existing calibration",
                f"The existing output file was not loaded:\n{exc}",
            )

    def display_to_original(self, x: float, y: float) -> tuple[float, float]:
        return x / self.scale, y / self.scale

    def original_to_display(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale, y * self.scale

    @staticmethod
    def parse_field_coordinate(text: str) -> tuple[float, float]:
        cleaned = text.replace("(", "").replace(")", "").strip()
        parts = [part.strip() for part in cleaned.split(",")]
        if len(parts) != 2:
            raise ValueError("Enter exactly two comma-separated numbers, e.g. 10, 23.583")
        return float(parts[0]), float(parts[1])

    def on_click(self, event: tk.Event) -> None:
        image_x, image_y = self.display_to_original(float(event.x), float(event.y))
        if not (0 <= image_x < self.original_w and 0 <= image_y < self.original_h):
            return

        prompt = (
            f"Image point: ({image_x:.1f}, {image_y:.1f}) px\n\n"
            "Enter the corresponding metric field coordinate as:\n"
            "field_x, field_y\n\n"
            "Both values are in yards. Example: 10, 23.583"
        )
        while True:
            value = simpledialog.askstring(
                "Field coordinate",
                prompt,
                parent=self.root,
            )
            if value is None:
                return
            try:
                field_x, field_y = self.parse_field_coordinate(value)
                break
            except ValueError as exc:
                messagebox.showerror("Invalid coordinate", str(exc), parent=self.root)

        self.points.append(
            {
                "image": [round(image_x, 4), round(image_y, 4)],
                "field": [field_x, field_y],
            }
        )
        self.redraw()

    def undo(self) -> None:
        if not self.points:
            return
        self.points.pop()
        self.redraw()

    def payload(self) -> Dict[str, object]:
        return {
            "video_id": self.video_id,
            "field_type": self.field_type,
            "image": str(self.image_path),
            "image_width": self.original_w,
            "image_height": self.original_h,
            "coordinate_convention": {
                "field_x": "longitudinal yards; consecutive yard lines differ by 5 yards",
                "field_y": "lateral yards; sideline-to-sideline direction",
            },
            "points": self.points,
        }

    def save(self, quiet: bool = False) -> bool:
        if len(self.points) < 4:
            if not quiet:
                messagebox.showwarning(
                    "Not enough points",
                    "A homography requires at least 4 non-collinear correspondences. "
                    "Please select at least 4 points; 6–8 are recommended.",
                    parent=self.root,
                )
            return False

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(self.payload(), indent=2) + "\n")
        self.status_var.set(f"{len(self.points)} points selected — saved to {self.output_path}")
        if not quiet:
            messagebox.showinfo(
                "Calibration saved",
                f"Saved {len(self.points)} correspondences to:\n{self.output_path}",
                parent=self.root,
            )
        return True

    def finish(self) -> None:
        if self.save(quiet=True):
            self.root.destroy()

    def on_close(self) -> None:
        if not self.points:
            self.root.destroy()
            return
        answer = messagebox.askyesnocancel(
            "Close calibrator",
            "Save the current correspondences before closing?",
            parent=self.root,
        )
        if answer is None:
            return
        if answer:
            if not self.save(quiet=True):
                return
        self.root.destroy()

    def redraw(self) -> None:
        self.canvas.delete("annotation")
        self.listbox.delete(0, tk.END)

        radius = 6
        for index, point in enumerate(self.points, start=1):
            image_x, image_y = point["image"]
            field_x, field_y = point["field"]
            dx, dy = self.original_to_display(float(image_x), float(image_y))
            self.canvas.create_oval(
                dx - radius,
                dy - radius,
                dx + radius,
                dy + radius,
                outline="red",
                width=2,
                tags=("annotation",),
            )
            self.canvas.create_text(
                dx + 10,
                dy - 10,
                text=str(index),
                fill="red",
                font=("TkDefaultFont", 11, "bold"),
                tags=("annotation",),
            )
            self.listbox.insert(
                tk.END,
                (
                    f"{index:02d}  img=({float(image_x):.1f}, {float(image_y):.1f})  "
                    f"field=({float(field_x):.3f}, {float(field_y):.3f})"
                ),
            )

        count = len(self.points)
        suffix = "point" if count == 1 else "points"
        quality_hint = "minimum reached" if count >= 4 else "need at least 4"
        if count >= 6:
            quality_hint = "recommended range reached"
        self.status_var.set(f"{count} {suffix} selected — {quality_hint}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Formation frame image (normally frame 0)")
    parser.add_argument("--output", type=Path, required=True, help="Output correspondence JSON")
    parser.add_argument("--video-id", required=True, help="Clip ID, e.g. JetSweep_2")
    parser.add_argument("--field-type", default="NFL", choices=("NFL", "NCAA", "NFHS"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise SystemExit(f"Image does not exist: {args.image}")
    if not args.image.is_file():
        raise SystemExit(f"Image path is not a file: {args.image}")

    root = tk.Tk()
    CalibrationGUI(
        root=root,
        image_path=args.image,
        output_path=args.output,
        video_id=args.video_id,
        field_type=args.field_type,
    )
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
