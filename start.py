import click
from VideoGenerator import live_video_generator


@click.command()
@click.option("--ext", "-e",
              default="png", type=str,
              help="extension of file to be rendered as rasengan. should be in [png,gif,mp4]", )
@click.option("--file", "-f", type=str,
              help="location of file to be renders as rasengan", )
def cli(ext: str, file: str) -> None:
    ext = ext.lower()
    if not ext in ['png', 'gif', 'mp4']:
        click.echo(f"not proper type of file")
        return
    click.echo(f"importing <{ext}> from <{file}>")
    live_video_generator(ext, file)
    click.echo("Done!")


if __name__ == "__main__":
    cli()