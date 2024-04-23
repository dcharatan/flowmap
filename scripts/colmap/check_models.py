from pathlib import Path

import click


@click.command()
@click.argument("input_path", type=click.Path(path_type=Path))
def main(input_path: Path) -> None:
    for scene in input_path.iterdir():
        try:
            # Check how many sparse models are in the scene.
            models = [x for x in (scene / "sparse").iterdir() if x.is_dir()]
            if len(models) > 1:
                # If there's more than one model, sparse reconstruction failed.
                print(f"{scene.name} has more than one model. Try re-running this.")
        except Exception:
            print(f"Something is wrong with {scene.name}. Investigate!")


if __name__ == "__main__":
    main()
