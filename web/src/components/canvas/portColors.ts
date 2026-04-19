/** Port-type → CSS color. Mirrors synapse/nodes/base.py::PORT_COLORS so the
 * web and desktop views stay visually consistent. Keep in sync manually — a
 * server endpoint could serve this eventually, but the color map is tiny
 * and rarely changes. */
export const PORT_COLORS: Record<string, string> = {
  table:       "rgb(52,152,219)",     // Blue
  stat:        "rgb(65,105,225)",     // RoyalBlue
  image:       "rgb(46,204,113)",     // Green
  mask:        "rgb(28,125,72)",      // Forest Green
  skeleton:    "rgb(180,230,100)",    // Yellow-green
  label:       "rgb(160,220,40)",     // Chartreuse
  label_image: "rgb(160,220,40)",     // Chartreuse alias
  figure:      "rgb(155,89,182)",     // Purple
  confocal:    "rgb(230,126,34)",     // Orange
  path:        "rgb(149,165,166)",    // Grey
  collection:  "rgb(230,180,50)",     // Gold
  model:       "rgb(255,140,66)",     // Coral/Orange
  html:        "rgb(235,87,135)",     // Rose/Pink
  any:         "rgb(95,106,106)",     // Dark grey
};

export function portColor(type: string): string {
  return PORT_COLORS[type] ?? PORT_COLORS.any;
}
