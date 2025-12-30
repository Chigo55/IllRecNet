#!/bin/bash

ruff check . --fix
ruff format .

pyrefly infer

pyrefly check