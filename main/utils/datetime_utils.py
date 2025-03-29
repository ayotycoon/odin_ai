from numpy.f2py.auxfuncs import throw_error
from pandas import Timedelta


def get_time_delta(time_str: str or None):
    if time_str is None:
        return None
    day_num = 0
    hour_num = 0
    min_num = 0
    sec_num = 0

    last_num = 0
    last_num_int = 0

    for c in time_str:
        if  c.isnumeric():
            last_num = last_num+ (int(c)* (last_num_int+1))
            last_num_int = last_num_int+1
        else:
            duration = c
            if duration == "D":
                day_num = last_num
            elif duration == "H":
                hour_num = last_num
            elif duration == "M":
                min_num = last_num
            elif duration == "S":
                sec_num = last_num


            last_num = 0
            last_num_int = 0



    return Timedelta(days=day_num,hours= hour_num, minutes=min_num, seconds=sec_num)