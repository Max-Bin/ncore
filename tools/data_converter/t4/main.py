# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T4 converter CLI entry point."""

from tools.data_converter.cli import cli
from tools.data_converter.t4.converter import t4_v4  # noqa: F401 -- registers CLI command


if __name__ == "__main__":
    cli(show_default=True)
