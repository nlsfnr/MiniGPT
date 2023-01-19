from pathlib import Path

from click.testing import CliRunner

from . import data


def test_cli_new_dataset(tmpdir: Path) -> None:
    runner = CliRunner()
    path = tmpdir / 'out'
    assert not path.exists()
    result = runner.invoke(data.get_cli(), ['new-dataset',
                                            '--path', str(path),
                                            '--name', 'dummy',
                                            '--min-length', '10',
                                            '--min-chars-per-token', '3'])
    assert result.exit_code == 0
    assert path.exists()
