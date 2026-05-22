#!/usr/bin/env python3
import sys

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('atk', 'def'):
        print('Usage: ros2 run strategy pk [atk|def]')
        sys.exit(1)

    mode = sys.argv.pop(1)  # remove before rclpy parses argv

    if mode == 'atk':
        from strategy.pk.pk_atk import main as atk_main
        atk_main()
    else:
        from strategy.pk.pk_def import main as def_main
        def_main()
