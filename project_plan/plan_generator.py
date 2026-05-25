import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# ============================================================
# CONFIGURATION
# ============================================================

EXCEL_FILE = "conversion_progress.xlsx"
OUTPUT_FILE = "conversion_dashboard.png"

# ============================================================
# STAGE CONFIGURATION
# ============================================================

stage_names = {
    1: "Executable Conversion",
    2: "Fidelity Alignment",
    3: "Performance Optimization",
    4: "Production Hardening"
}

stage_short_names = {
    1: "Exec",
    2: "Fidelity",
    3: "Performance",
    4: "Production"
}

stage_colors = {
    1: "#1565C0",   # Material Deep Blue
    2: "#2E7D32",   # Material Deep Green
    3: "#E65100",   # Material Deep Orange
    4: "#6A1B9A"    # Material Deep Purple
}

confidence_colors = {
    "High": "#2E7D32",
    "Medium": "#E65100",
    "Low": "#C62828"
}

# ============================================================
# PROGRESS MAPPING
# ============================================================

stage_progress = {
    1: 25,
    2: 50,
    3: 75,
    4: 100
}

# ============================================================
# LOAD DATA
# ============================================================

def load_data(excel_file):

    modules_df = pd.read_excel(
        excel_file,
        sheet_name="Modules"
    )

    utilities_df = pd.read_excel(
        excel_file,
        sheet_name="Utilities"
    )

    return modules_df, utilities_df

# ============================================================
# DRAW MODULE SECTION
# ============================================================

def draw_module_section(ax, modules_df):

    # ========================================================
    # MODULE COUNT
    # ========================================================

    num_modules = len(modules_df)

    # ========================================================
    # AXIS SETUP
    # ========================================================

    ax.set_xlim(0, 8)
    ax.set_ylim(0, num_modules + 1.8)

    ax.axis('off')

    # ========================================================
    # SHADED SECTION HEADER
    # ========================================================

    header_y = num_modules + 1.35

    # Background Banner
    ax.add_patch(
        Rectangle(
            (0, header_y - 0.18),
            8.0,
            0.42,
            color="#1565C0",
            alpha=1.0,
            zorder=0
        )
    )

    # Title
    ax.text(
        0.2,
        header_y,
        "MODULE CONVERSION PROGRESS",
        fontsize=16,
        fontweight='bold',
        color="white",
        va='center',
        ha='left'
    )

    # ========================================================
    # STAGE HEADERS
    # ========================================================

    for i in range(4):

        x = i + 1

        ax.add_patch(
            Rectangle(
                (x, num_modules + 0.3),
                1,
                0.8,
                color=stage_colors[i + 1],
                alpha=0.9
            )
        )

        ax.text(
            x + 0.5,
            num_modules + 0.7,
            f"Stage {i+1}\n{stage_short_names[i+1]}",
            ha='center',
            va='center',
            fontsize=10,
            color='white',
            fontweight='bold'
        )

    # ========================================================
    # COLUMN HEADERS
    # ========================================================

    ax.text(
        5.55,
        num_modules + 0.7,
        "Current\nStage",
        ha='center',
        va='center',
        fontsize=10,
        fontweight='bold'
    )

    ax.text(
        6.6,
        num_modules + 0.7,
        "Overall\nProgress",
        ha='center',
        va='center',
        fontsize=10,
        fontweight='bold'
    )

    # ========================================================
    # DRAW MODULE ROWS
    # ========================================================

    for idx, row in modules_df.iterrows():

        y = num_modules - idx - 0.2

        module = row["Module"]
        stage = int(row["Stage"])
        confidence = row["Confidence"]

        progress = stage_progress[stage]

        # ----------------------------------------------------
        # Alternating Row Background
        # ----------------------------------------------------

        row_bg = "#EEF2F7" if idx % 2 == 0 else "#FFFFFF"
        ax.add_patch(
            Rectangle(
                (0.0, y - 0.28),
                8.0,
                0.56,
                color=row_bg,
                alpha=1.0,
                zorder=0
            )
        )
        # Bottom divider line
        ax.plot(
            [0.0, 7.8],
            [y - 0.28, y - 0.28],
            color="#D0D7E2",
            linewidth=0.6,
            zorder=1
        )

        # ----------------------------------------------------
        # Module Name
        # ----------------------------------------------------

        ax.text(
            0.1,
            y,
            module,
            fontsize=11,
            fontweight='bold',
            va='center',
            color='#1A2A3D'
        )

        # ----------------------------------------------------
        # Stage Progression
        # ----------------------------------------------------

        for s in range(1, 5):

            if s <= stage:
                color = stage_colors[s]
                alpha = 1.0
            else:
                color = "#D5D8DC"
                alpha = 0.4

            ax.plot(
                [s + 0.1, s + 0.9],
                [y, y],
                color=color,
                linewidth=8,
                alpha=alpha,
                solid_capstyle='round'
            )

            ax.scatter(
                s + 0.5,
                y,
                s=300,
                color=color,
                edgecolors='black',
                zorder=3
            )

        # ----------------------------------------------------
        # Current Stage Badge
        # ----------------------------------------------------

        badge_x = 5.55

        ax.add_patch(
            Rectangle(
                (badge_x - 0.3, y - 0.16),
                0.6,
                0.32,
                color=stage_colors[stage],
                alpha=0.95
            )
        )

        ax.text(
            badge_x,
            y,
            f"Stage {stage}",
            fontsize=9,
            fontweight='bold',
            color='white',
            ha='center',
            va='center'
        )

        # ----------------------------------------------------
        # Overall Progress Bar
        # ----------------------------------------------------

        progress_x = 6.6

        # Background
        ax.add_patch(
            Rectangle(
                (progress_x - 0.3, y - 0.10),
                0.6,
                0.20,
                color="#E0E6EF",
                zorder=2
            )
        )

        # Fill
        ax.add_patch(
            Rectangle(
                (progress_x - 0.3, y - 0.10),
                0.6 * (progress / 100),
                0.20,
                color=stage_colors[stage],
                zorder=3
            )
        )

        ax.text(
            progress_x + 0.4,
            y,
            f"{progress}%",
            fontsize=10,
            fontweight='bold',
            va='center',
            color=stage_colors[stage]
        )

# ============================================================
# DRAW UTILITIES SECTION
# ============================================================

def draw_utilities_section(ax, utilities_df):

    ax.axis('off')

    # ========================================================
    # SHADED SECTION HEADER
    # ========================================================

    ax.add_patch(
        Rectangle(
            (0.0, 1.01),
            1.0,
            0.10,
            color="#6A1B9A",
            alpha=1.0,
            transform=ax.transAxes,
            clip_on=False
        )
    )

    ax.text(
        0.02,
        1.06,
        "SHARED UTILITIES PROGRESS (PLATFORM TRACK)",
        fontsize=16,
        fontweight='bold',
        color="white",
        va='center',
        ha='left',
        transform=ax.transAxes
    )

    # ========================================================
    # COLUMN HEADERS
    # ========================================================

    headers = [
        ("Key Utilities", 0.02),
        ("Examples", 0.22),
        ("Progress", 0.42),
        ("Status", 0.72),
        ("Details", 0.84)
    ]

    for header, x in headers:

        ax.text(
            x,
            0.95,
            header,
            fontsize=11,
            fontweight='bold',
            color='#1A2A3D',
            transform=ax.transAxes
        )

    # ========================================================
    # ROWS
    # ========================================================

    row_height = 0.17

    for idx, row in utilities_df.iterrows():

        utility_group = row["Utility Group"]
        examples = row["Examples"]
        progress = row["Progress"]
        status = row["Status"]
        details = row["Details"]

        y = 0.82 - idx * row_height

        # ----------------------------------------------------
        # Alternating Row Background
        # ----------------------------------------------------

        bg_color = "#EEF2F7" if idx % 2 == 0 else "#FFFFFF"

        ax.add_patch(
            Rectangle(
                (0.01, y - 0.06),
                0.98,
                0.12,
                color=bg_color,
                transform=ax.transAxes,
                zorder=0
            )
        )
        # Bottom divider
        ax.plot(
            [0.01, 0.99],
            [y - 0.06, y - 0.06],
            color="#D0D7E2",
            linewidth=0.6,
            transform=ax.transAxes,
            zorder=1
        )

        # ----------------------------------------------------
        # Utility Group + Examples
        # ----------------------------------------------------

        ax.text(
            0.02,
            y,
            utility_group,
            fontsize=10,
            fontweight='bold',
            va='center',
            transform=ax.transAxes
        )

        ax.text(
            0.22,
            y,
            examples,
            fontsize=9,
            va='center',
            color='#4A5568',
            transform=ax.transAxes
        )

        # ----------------------------------------------------
        # Progress Bar Background
        # ----------------------------------------------------

        ax.add_patch(
            Rectangle(
                (0.42, y - 0.02),
                0.22,
                0.04,
                color="#E0E6EF",
                transform=ax.transAxes
            )
        )

        # ----------------------------------------------------
        # Progress Bar Fill
        # ----------------------------------------------------

        if progress >= 75:
            progress_color = "#2E7D32"
        elif progress >= 50:
            progress_color = "#E65100"
        else:
            progress_color = "#1565C0"

        ax.add_patch(
            Rectangle(
                (0.42, y - 0.02),
                0.22 * (progress / 100),
                0.04,
                color=progress_color,
                transform=ax.transAxes
            )
        )

        # ----------------------------------------------------
        # Progress %
        # ----------------------------------------------------

        ax.text(
            0.65,
            y,
            f"{progress}%",
            fontsize=10,
            fontweight='bold',
            color=progress_color,
            va='center',
            transform=ax.transAxes
        )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if status.lower() == "complete":
            status_color = "#27AE60"
        elif status.lower() == "in progress":
            status_color = "#F39C12"
        else:
            status_color = "#5D6D7E"

        ax.text(
            0.72,
            y,
            status,
            fontsize=10,
            fontweight='bold',
            color=status_color,
            va='center',
            transform=ax.transAxes
        )

        # ----------------------------------------------------
        # Details
        # ----------------------------------------------------

        ax.text(
            0.84,
            y,
            details,
            fontsize=9,
            va='center',
            transform=ax.transAxes
        )

# ============================================================
# ADD LEGEND
# ============================================================

def add_legend(fig):

    legend_elements = [
        Line2D(
            [0],
            [0],
            color=stage_colors[1],
            lw=6,
            label='Stage 1 - Executable Conversion'
        ),

        Line2D(
            [0],
            [0],
            color=stage_colors[2],
            lw=6,
            label='Stage 2 - Fidelity Alignment'
        ),

        Line2D(
            [0],
            [0],
            color=stage_colors[3],
            lw=6,
            label='Stage 3 - Performance Optimization'
        ),

        Line2D(
            [0],
            [0],
            color=stage_colors[4],
            lw=6,
            label='Stage 4 - Production Hardening'
        )
    ]

    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=2,
        fontsize=10
    )

# ============================================================
# SAVE CHART
# ============================================================

def save_chart(fig, output_file):

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches='tight'
    )

    print(f"\nChart saved to: {output_file}")

# ============================================================
# GENERATE DASHBOARD
# ============================================================

def generate_dashboard(excel_file, output_file):

    # --------------------------------------------------------
    # Load Data
    # --------------------------------------------------------

    modules_df, utilities_df = load_data(excel_file)

    # --------------------------------------------------------
    # Create Figure
    # --------------------------------------------------------

    fig = plt.figure(figsize=(20, 12), facecolor='#F5F7FA')

    gs = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[3, 2]
    )

    ax_modules = fig.add_subplot(gs[0])
    ax_modules.set_facecolor('#F5F7FA')
    ax_utils = fig.add_subplot(gs[1])
    ax_utils.set_facecolor('#F5F7FA')

    # --------------------------------------------------------
    # Dashboard Title
    # --------------------------------------------------------

    fig.suptitle(
        "COBOL CONVERSION PROGRAM - OVERALL PROGRESS",
        fontsize=24,
        fontweight='bold',
        color='#1A2A3D'
    )

    # --------------------------------------------------------
    # Draw Sections
    # --------------------------------------------------------

    draw_module_section(ax_modules, modules_df)

    draw_utilities_section(ax_utils, utilities_df)

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # --------------------------------------------------------
    # Save Chart
    # --------------------------------------------------------

    save_chart(fig, output_file)

    # --------------------------------------------------------
    # Show Chart
    # --------------------------------------------------------

    plt.show()

# ============================================================
# EXECUTE
# ============================================================

if __name__ == "__main__":

    generate_dashboard(
        EXCEL_FILE,
        OUTPUT_FILE
    )