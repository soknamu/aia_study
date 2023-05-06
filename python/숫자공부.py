# def count_values(filename, start, end, step):
#     """
#     주어진 파일(filename)에서 start와 end 범위 사이의 step 간격에 따른
#     값을 구하고, 해당 범위 안에 속하는 값의 개수를 세는 함수
#     """
#     def generate_ranges(start, end, step):
#         """
#         주어진 범위(start~end)와 간격(step)에 따라 숫자의 범위를 생성하는 함수
#         """
#         ranges = []
#         while start < end:
#             ranges.append((round(start, 2), round(start+step, 2)))
#             start += step
#         return ranges

#     # 주어진 범위에 해당하는 값을 구하기 위한 범위 생성
#     ranges = generate_ranges(start, end, step)

#     # 주어진 파일에서 해당 범위 안에 속하는 값을 찾아 개수를 세기 위한 딕셔너리 생성
#     counts = {range_: 0 for range_ in ranges}

#     with open(filename, 'r') as f:
#         # 첫번째 줄은 header 이므로 skip
#         next(f)
#         for line in f:
#             # 콤마로 구분된 각 필드에 대해 loop
#             for value in line.strip().split(','):
#                 try:
#                     # 문자열을 실수형으로 변환
#                     value_float = float(value)
#                     # 주어진 범위 안에 속하는지 확인하고 딕셔너리의 값을 증가
#                     for range_ in ranges:
#                         if range_[0] <= value_float <= range_[1]:
#                             counts[range_] += 1
#                 except ValueError:
#                     # 문자열을 실수형으로 변환할 수 없는 경우 무시
#                     pass

#     return counts

def count_values(filename, start, end, step):
    """
    주어진 파일(filename)에서 start와 end 범위 사이의 step 간격에 따른
    값을 구하고, 해당 범위 안에 속하는 값의 개수를 세는 함수
    """
    ranges = []
    current = round(start, 2)
    while current < end:
        ranges.append((current, round(current+step, 2)))
        current += step

    counts = {range_: 0 for range_ in ranges}

    with open(filename, 'r') as f:
        next(f)
        for line in f:
            for value in line.strip().split(','):
                try:
                    value_float = float(value)
                    for range_ in ranges:
                        if range_[0] <= value_float <= range_[1]:
                            counts[range_] += 1
                except ValueError:
                    pass

    return counts


counts = count_values('c:/study/_save/dacon_airplaneX/x_submission_678진짜해보기.csv', 0.5, 0.8, 0.01)
# counts = count_values('c:/study/_save/dacon_airplaneX/데이콘/제출용10.csv', 0.5, 0.8, 0.01)

# 각 범위에 속하는 값의 개수 출력
for range_, count in counts.items():
    print(f"{range_}: {count}")