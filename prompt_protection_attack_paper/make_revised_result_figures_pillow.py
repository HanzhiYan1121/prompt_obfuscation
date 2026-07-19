from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


METHOD_LABELS = [
    "ConfusionPrompt",
    "InferDPT",
    "Split-N-Denoise",
    "Confidential Prompting",
    "NOIR",
    "EmojiPrompt",
    "DP-OPT",
    "Prεεmpt",
]
METHOD_LABELS_WRAPPED = [
    "ConfusionPrompt",
    "InferDPT",
    "Split-N-Denoise",
    "Confidential\nPrompting",
    "NOIR",
    "EmojiPrompt",
    "DP-OPT",
    "Prεεmpt",
]

TEXT_F1 = [0.667, 0.671, 0.380, 0.481, 0.389, 0.546, 0.444, 0.679]
ROUGE_F1 = [0.596, 0.588, 0.312, 0.391, 0.314, 0.477, 0.353, 0.606]
UTILITY_F1 = [0.406, 0.433, 0.366, 0.388, 0.387, 0.391, 0.361, 0.425]

CATEGORIES = ["Role", "Task", "Format", "Constraints", "Context", "Tool"]
UTILITY_GRID = [
    [0.479, 0.431, 0.396, 0.327, 0.432, 0.370],
    [0.495, 0.444, 0.447, 0.400, 0.451, 0.358],
    [0.444, 0.392, 0.327, 0.292, 0.387, 0.354],
    [0.445, 0.424, 0.383, 0.340, 0.374, 0.361],
    [0.461, 0.405, 0.374, 0.359, 0.366, 0.355],
    [0.459, 0.398, 0.380, 0.335, 0.414, 0.360],
    [0.427, 0.353, 0.328, 0.309, 0.398, 0.351],
    [0.478, 0.441, 0.466, 0.347, 0.453, 0.362],
]

FONT_REGULAR = Path(r"C:\Windows\Fonts\arial.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(str(path), size=size)


def centered_text(draw: ImageDraw.ImageDraw, box, text: str, text_font, fill) -> None:
    left, top, right, bottom = box
    bounds = draw.multiline_textbbox((0, 0), text, font=text_font, spacing=4, align="center")
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.multiline_text(
        ((left + right - width) / 2, (top + bottom - height) / 2 - bounds[1]),
        text,
        font=text_font,
        fill=fill,
        spacing=4,
        align="center",
    )


def make_average_figure(output_dir: Path) -> None:
    width, height = 2500, 1180
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    left, right, top, bottom = 175, 70, 170, 260
    plot_left, plot_right = left, width - right
    plot_top, plot_bottom = top, height - bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top
    maximum = 0.75

    for tick in range(0, 8):
        value = tick / 10
        y = plot_bottom - value / maximum * plot_height
        draw.line((plot_left, y, plot_right, y), fill="#d7dee7", width=2)
        label = f"{value:.1f}"
        bounds = draw.textbbox((0, 0), label, font=font(28))
        draw.text((plot_left - 22 - (bounds[2] - bounds[0]), y - 15), label, fill="#2d3748", font=font(28))

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#2d3748", width=3)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#2d3748", width=3)

    series = [
        ("Text F1", TEXT_F1, "#2b6cb0"),
        ("ROUGE-L F1", ROUGE_F1, "#63b3ed"),
        ("Utility F1", UTILITY_F1, "#dd6b20"),
    ]
    group_width = plot_width / len(METHOD_LABELS)
    bar_width = 58
    bar_gap = 8
    bars_width = len(series) * bar_width + (len(series) - 1) * bar_gap

    for method_index, label in enumerate(METHOD_LABELS_WRAPPED):
        group_left = plot_left + method_index * group_width
        bars_left = group_left + (group_width - bars_width) / 2
        for series_index, (_, values, color) in enumerate(series):
            value = values[method_index]
            x0 = bars_left + series_index * (bar_width + bar_gap)
            y0 = plot_bottom - value / maximum * plot_height
            draw.rectangle((x0, y0, x0 + bar_width, plot_bottom), fill=color)
        centered_text(
            draw,
            (group_left + 3, plot_bottom + 18, group_left + group_width - 3, height - 25),
            label,
            font(27),
            "#1a202c",
        )

    legend_x = 620
    for label, _, color in series:
        draw.rounded_rectangle((legend_x, 58, legend_x + 44, 102), radius=4, fill=color)
        draw.text((legend_x + 58, 61), label, font=font(31), fill="#1a202c")
        label_width = draw.textbbox((0, 0), label, font=font(31))[2]
        legend_x += 58 + label_width + 75

    y_label = Image.new("RGBA", (320, 70), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_label)
    y_draw.text((0, 5), "Mean score", font=font(31, bold=True), fill="#1a202c")
    y_label = y_label.rotate(90, expand=True)
    image.paste(y_label, (30, (plot_top + plot_bottom - y_label.height) // 2), y_label)

    image.save(output_dir / "privacybench8_method_average_revised.png", dpi=(300, 300))


def blue_scale(value: float) -> tuple[int, int, int]:
    minimum, maximum = 0.28, 0.50
    ratio = max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))
    light = (239, 243, 255)
    dark = (8, 81, 156)
    return tuple(round(light[i] + ratio * (dark[i] - light[i])) for i in range(3))


def luminance(color: tuple[int, int, int]) -> float:
    red, green, blue = [component / 255 for component in color]
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def make_heatmap(output_dir: Path) -> None:
    width, height = 2400, 1380
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    left, right, top, bottom = 520, 265, 140, 130
    grid_left, grid_right = left, width - right
    grid_top, grid_bottom = top, height - bottom
    cell_width = (grid_right - grid_left) / len(CATEGORIES)
    cell_height = (grid_bottom - grid_top) / len(METHOD_LABELS)

    for column, label in enumerate(CATEGORIES):
        centered_text(
            draw,
            (grid_left + column * cell_width, 20, grid_left + (column + 1) * cell_width, grid_top - 8),
            label,
            font(31, bold=True),
            "#1a202c",
        )

    for row, method in enumerate(METHOD_LABELS):
        row_top = grid_top + row * cell_height
        centered_text(draw, (15, row_top, grid_left - 28, row_top + cell_height), method, font(31), "#1a202c")
        for column, value in enumerate(UTILITY_GRID[row]):
            cell_left = grid_left + column * cell_width
            cell_top = row_top
            color = blue_scale(value)
            draw.rectangle(
                (cell_left, cell_top, cell_left + cell_width, cell_top + cell_height),
                fill=color,
                outline="white",
                width=3,
            )
            annotation_color = "black" if luminance(color) > 0.56 else "white"
            centered_text(
                draw,
                (cell_left, cell_top, cell_left + cell_width, cell_top + cell_height),
                f"{value:.3f}",
                font(34, bold=True),
                annotation_color,
            )

    bar_left = width - 170
    bar_top = grid_top
    bar_bottom = grid_bottom
    for y in range(int(bar_top), int(bar_bottom)):
        ratio = 1 - (y - bar_top) / (bar_bottom - bar_top)
        value = 0.28 + ratio * (0.50 - 0.28)
        draw.line((bar_left, y, bar_left + 48, y), fill=blue_scale(value), width=2)
    draw.rectangle((bar_left, bar_top, bar_left + 48, bar_bottom), outline="#4a5568", width=2)

    for value in [0.28, 0.35, 0.40, 0.45, 0.50]:
        y = bar_bottom - (value - 0.28) / (0.50 - 0.28) * (bar_bottom - bar_top)
        draw.line((bar_left + 48, y, bar_left + 62, y), fill="#2d3748", width=2)
        draw.text((bar_left + 70, y - 17), f"{value:.2f}", font=font(27), fill="#2d3748")

    image.save(output_dir / "privacybench8_utility_heatmap_revised.png", dpi=(300, 300))


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    make_average_figure(output_dir)
    make_heatmap(output_dir)


if __name__ == "__main__":
    main()
