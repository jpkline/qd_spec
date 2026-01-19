# Copyright 2026 John Kline

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd


def save_temp(result):
    try:
        df = pd.read_csv("temp_results.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame({k: v.value for k, v in result.params.items()}, index=[0])
    else:
        df = pd.concat([df, pd.DataFrame({k: v.value for k, v in result.params.items()}, index=[0])])
    df.to_csv("temp_results.csv", index=False)
