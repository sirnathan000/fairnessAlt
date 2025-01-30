#!/bin/bash

find "results/student/mondrian" -type f -exec rm -f {} +
find "results/student/modified_mondrian" -type f -exec rm -f {} +



python3 anonymize.py --method=mondrian --k=2 --dataset=student
python3 anonymize.py --method=mondrian --k=3 --dataset=student
python3 anonymize.py --method=mondrian --k=4 --dataset=student
python3 anonymize.py --method=mondrian --k=5 --dataset=student
python3 anonymize.py --method=mondrian --k=6 --dataset=student
python3 anonymize.py --method=mondrian --k=7 --dataset=student
python3 anonymize.py --method=mondrian --k=8 --dataset=student
python3 anonymize.py --method=mondrian --k=9 --dataset=student
python3 anonymize.py --method=mondrian --k=10 --dataset=student
python3 anonymize.py --method=mondrian --k=25 --dataset=student
python3 anonymize.py --method=mondrian --k=50 --dataset=student
python3 anonymize.py --method=mondrian --k=75 --dataset=student
python3 anonymize.py --method=mondrian --k=100 --dataset=student

python3 anonymize.py --method=modified_mondrian --k=2 --dataset=student
python3 anonymize.py --method=modified_mondrian --k=3 --dataset=student
python3 anonymize.py --method=modified_mondrian --k=4 --dataset=student
python3 anonymize.py --method=modified_mondrian --k=5 --dataset=student
python3 anonymize.py --method=modified_mondrian --k=6 --dataset=student
python3 anonymize.py --method=modified_mondrian --k=7 --dataset=student
python3 anonymize.py --method=modified_mondrian --k=8 --dataset=student
python3 anonymize.py --method=modified_mondrian --k=9 --dataset=student
python3 anonymize.py --method=modified_mondrian --k=10 --dataset=student
python3 anonymize.py --method=modified_mondrian --k=25 --dataset=student
python3 anonymize.py --method=modified_mondrian --k=50 --dataset=student
python3 anonymize.py --method=modified_mondrian --k=75 --dataset=student
python3 anonymize.py --method=modified_mondrian --k=100 --dataset=student


