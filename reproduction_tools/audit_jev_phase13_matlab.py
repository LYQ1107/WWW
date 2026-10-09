"""Inspect supplied MATLAB media without installing or exposing license files."""
from datetime import datetime, timezone
import shutil
import xml.etree.ElementTree as ET
from jev_phase13_common import *


def main():
    report_path = REPORTS / 'MATLAB_ENVIRONMENT_AUDIT.json'
    previous = json.loads(report_path.read_text()) if report_path.exists() else None
    media = Path('/home/liuyeqiang/2020a/Matlab908Lin')
    iso = media / 'Matlab98R2020a_Lin64.iso'
    version_xml = subprocess.check_output(
        ['7z', 'e', '-so', str(iso), 'VersionInfo.xml'], text=True)
    version = ET.fromstring(version_xml)
    scopes = ['/home/liuyeqiang', '/opt', '/usr/local', '/data1/liuyeqiang']
    search = subprocess.run(
        ['find', *scopes, '-type', 'f', '(', '-path', '*/bin/matlab',
         '-o', '-name', 'MATLAB', '-o', '-name', 'octave-cli', ')', '-print'],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    executables = search.stdout.splitlines()
    runtime_found = bool(shutil.which('matlab') or executables)
    report = {
        'status': 'RUNTIME_CANDIDATES_FOUND' if runtime_found else 'INSTALL_MEDIA_ONLY',
        'UTC': datetime.now(timezone.utc).isoformat(),
        'media_directory': str(media), 'ISO': str(iso),
        'ISO_bytes': iso.stat().st_size,
        'media_release': version.findtext('release'),
        'media_version': version.findtext('version'),
        'media_VersionInfo_xml': version_xml,
        'mount_directory_empty': not any((media / 'mnt').iterdir()),
        'PATH_matlab': shutil.which('matlab'), 'PATH_octave': shutil.which('octave'),
        'searched_roots': scopes, 'executable_candidates': executables,
        'search_exit_code': search.returncode,
        'unreadable_search_entries': len(search.stderr.splitlines()),
        'search_scope': 'accessible files under listed roots; not a claim about every user or device',
        'license_validation': 'NOT_RUN: no runnable installation found; license files not read or modified',
        'official_toolkit': str(ROOT / 'MOTChallengeEvalKit_cv_test/matlab_devkit'),
        'official_MATLAB_evaluation': {
            'status': 'NOT_RUN', 'metrics': None,
            'reason': 'R2020a installation ISO is present, but no MATLAB executable was found in the inspected roots or PATH; native official evaluator requires an installed runnable environment'},
        'existing_training_or_evaluation_interrupted': False,
    }
    installed = Path('/data1/liuyeqiang/MATLAB/R2020a')
    smoke = Path('/data1/liuyeqiang/matlab_R2020a_install_tools/official_metric_smoke.json')
    if smoke.exists():
        actual = json.loads(smoke.read_text())
        assert actual['status'] == 'PASS'
        report.update(status='NATIVE_MATLAB_READY', installed_root=str(installed),
                      native_version=actual['version'], native_release=actual['release'],
                      official_metric_smoke={'path':str(smoke),'SHA256':sha(smoke),'values':actual},
                      license_validation='native headless MATLAB startup and official MEX computations completed successfully')
        report['official_MATLAB_evaluation']['reason'] = 'native R2020a and official metric smoke passed; actual frozen-video evaluations pending'
        report['initial_observation'] = (previous.get('initial_observation',previous)
                                         if previous is not None else None)
    save(report_path, report)
    print('PHASE13_MATLAB_AUDIT', report['status'], report['media_release'],
          'executables', executables, flush=True)


if __name__ == '__main__':
    main()
