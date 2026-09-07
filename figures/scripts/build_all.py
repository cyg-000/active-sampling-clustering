"""Rebuild the five final manuscript figures from the workspace data."""
import build_aperture_five_step
import build_figure1_final
import build_figure2_concept
import build_revised_five
import build_figure4_revised
import build_figure5_revised


def main():
    build_aperture_five_step.main()
    build_figure1_final.main()
    build_figure2_concept.main()
    # Historical function name retained in the source; it writes Figure 3.
    build_revised_five.build_figure2()
    build_figure4_revised.main()
    build_figure5_revised.main()


if __name__ == "__main__":
    main()
