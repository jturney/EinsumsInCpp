#!/usr/bin/env python3
"""
Post-process HTML files output from Sphinx.
"""
import io

def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', help='file mode', choices=('html',))
    parser.add_argument('file', nargs='+', help='input file(s)')
    args = parser.parse_args()

    mode = args.mode

    for fn in args.file:
        with io.open(fn, 'r', encoding='utf-8') as f:
            if mode == 'html':
                lines = process_html(fn, f.readlines())

        with io.open(fn, 'w', encoding='utf-8') as f:
            f.write(''.join(lines))

def process_html(fn, lines):
    return lines

if __name__ == "__main__":
    main()
