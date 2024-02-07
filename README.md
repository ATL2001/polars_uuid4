## Polars fast UUID4 string generation


```python
import polars as pl
import polars.selectors as cs
import numpy as np
import uuid

import polars_uuid4
```


```python
pl.__version__
```




    '0.20.7'



##### Make dataframe of with 10 million random numbers


```python
df = pl.DataFrame({
    'Random numbers': np.random.rand(10000000),
    'A string column': "value",
}).with_row_index()
df.tail()
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (5, 3)</small><table border="1" class="dataframe"><thead><tr><th>index</th><th>Random numbers</th><th>A string column</th></tr><tr><td>u32</td><td>f64</td><td>str</td></tr></thead><tbody><tr><td>9999995</td><td>0.342875</td><td>&quot;value&quot;</td></tr><tr><td>9999996</td><td>0.283626</td><td>&quot;value&quot;</td></tr><tr><td>9999997</td><td>0.91639</td><td>&quot;value&quot;</td></tr><tr><td>9999998</td><td>0.299616</td><td>&quot;value&quot;</td></tr><tr><td>9999999</td><td>0.460211</td><td>&quot;value&quot;</td></tr></tbody></table></div>



##### Create 10 million UUID4s
 * with_uuid4() accepts a variable so you can set the name of the series, defaults to uuid


```python
df.uuid.with_uuid4()
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (10_000_000, 4)</small><table border="1" class="dataframe"><thead><tr><th>index</th><th>Random numbers</th><th>A string column</th><th>uuid</th></tr><tr><td>u32</td><td>f64</td><td>str</td><td>str</td></tr></thead><tbody><tr><td>0</td><td>0.431903</td><td>&quot;value&quot;</td><td>&quot;{57cfa3fd-01a5…</td></tr><tr><td>1</td><td>0.198707</td><td>&quot;value&quot;</td><td>&quot;{3e418a42-db42…</td></tr><tr><td>2</td><td>0.626431</td><td>&quot;value&quot;</td><td>&quot;{1e16aeb2-0675…</td></tr><tr><td>3</td><td>0.790102</td><td>&quot;value&quot;</td><td>&quot;{e1129c0a-38e1…</td></tr><tr><td>4</td><td>0.907382</td><td>&quot;value&quot;</td><td>&quot;{8ad58341-ab23…</td></tr><tr><td>5</td><td>0.995303</td><td>&quot;value&quot;</td><td>&quot;{83ed9d53-30a5…</td></tr><tr><td>6</td><td>0.998931</td><td>&quot;value&quot;</td><td>&quot;{2ce35a0f-9981…</td></tr><tr><td>7</td><td>0.836289</td><td>&quot;value&quot;</td><td>&quot;{655d0891-0f1b…</td></tr><tr><td>8</td><td>0.872352</td><td>&quot;value&quot;</td><td>&quot;{77fec4e7-1a23…</td></tr><tr><td>9</td><td>0.529137</td><td>&quot;value&quot;</td><td>&quot;{912c7ff7-0a12…</td></tr><tr><td>10</td><td>0.322931</td><td>&quot;value&quot;</td><td>&quot;{d6402d0d-b5ab…</td></tr><tr><td>11</td><td>0.456256</td><td>&quot;value&quot;</td><td>&quot;{26c89cc9-d740…</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>9999988</td><td>0.006378</td><td>&quot;value&quot;</td><td>&quot;{ddc657e6-2fa7…</td></tr><tr><td>9999989</td><td>0.50514</td><td>&quot;value&quot;</td><td>&quot;{3a7f87a4-23de…</td></tr><tr><td>9999990</td><td>0.708277</td><td>&quot;value&quot;</td><td>&quot;{b51b0665-32a0…</td></tr><tr><td>9999991</td><td>0.743679</td><td>&quot;value&quot;</td><td>&quot;{5fe2070b-9d4c…</td></tr><tr><td>9999992</td><td>0.937289</td><td>&quot;value&quot;</td><td>&quot;{11b6f029-6d44…</td></tr><tr><td>9999993</td><td>0.763785</td><td>&quot;value&quot;</td><td>&quot;{44b87135-d0a7…</td></tr><tr><td>9999994</td><td>0.913705</td><td>&quot;value&quot;</td><td>&quot;{9127c91c-2a4f…</td></tr><tr><td>9999995</td><td>0.342875</td><td>&quot;value&quot;</td><td>&quot;{4dcc6d5e-97da…</td></tr><tr><td>9999996</td><td>0.283626</td><td>&quot;value&quot;</td><td>&quot;{3b34e5ff-1047…</td></tr><tr><td>9999997</td><td>0.91639</td><td>&quot;value&quot;</td><td>&quot;{d32b1a17-50ba…</td></tr><tr><td>9999998</td><td>0.299616</td><td>&quot;value&quot;</td><td>&quot;{71ad3545-fe92…</td></tr><tr><td>9999999</td><td>0.460211</td><td>&quot;value&quot;</td><td>&quot;{5ca39c0a-9993…</td></tr></tbody></table></div>



#### Works with a lazy frame too


```python
df = pl.LazyFrame({
    'Random numbers': np.random.rand(10000000),
    'A string column': "value",
}).with_row_index().uuid.with_uuid4().collect()
df.tail()
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (5, 4)</small><table border="1" class="dataframe"><thead><tr><th>index</th><th>Random numbers</th><th>A string column</th><th>uuid</th></tr><tr><td>u32</td><td>f64</td><td>str</td><td>str</td></tr></thead><tbody><tr><td>9999995</td><td>0.185959</td><td>&quot;value&quot;</td><td>&quot;{c4baf1ce-98c5…</td></tr><tr><td>9999996</td><td>0.005801</td><td>&quot;value&quot;</td><td>&quot;{172ddf3c-ea9b…</td></tr><tr><td>9999997</td><td>0.606094</td><td>&quot;value&quot;</td><td>&quot;{3dc75c0d-19fd…</td></tr><tr><td>9999998</td><td>0.268984</td><td>&quot;value&quot;</td><td>&quot;{f9a4f709-a2e9…</td></tr><tr><td>9999999</td><td>0.22677</td><td>&quot;value&quot;</td><td>&quot;{75f6c83d-a693…</td></tr></tbody></table></div>



##### My old way to generate a UUID4 for each row
  * Gets job done.  Creates a UUID4 for each row.
  * Uses python uuid module.
  * Takes a long time (in the polars world).
    * 20.7 s ± 91 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


```python
%%timeit
uuids = ["{"+str(uuid.uuid4())+"}" for i in range(len(df))]
uuid_series = pl.Series(name="python_UUID", values=uuids)
df.with_columns(
    uuid_series
)
```

    20.4 s ± 160 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    

##### Using pl_uuid to generate a UUID4 for each row 
  * Gets job done.  Creates a UUID4 for each row.
  * Uses rust uuid crate.
  * Much easier to understand/simpler code.
  * ~ 40x faster than using python's uuid module to generate UUID4 when the last column in the df is already a string
  * 512 ms ± 6.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


```python
%%timeit
df.uuid.with_uuid4()
```

    512 ms ± 6.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    

##### Not quite as fast if there isnt an existing string column in the dataframe
  * 644 ms ± 6.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


```python
df = pl.DataFrame({
    'Random numbers': np.random.rand(10000000),
}).with_row_index()
df.tail()
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (5, 2)</small><table border="1" class="dataframe"><thead><tr><th>index</th><th>Random numbers</th></tr><tr><td>u32</td><td>f64</td></tr></thead><tbody><tr><td>9999995</td><td>0.313362</td></tr><tr><td>9999996</td><td>0.679717</td></tr><tr><td>9999997</td><td>0.076164</td></tr><tr><td>9999998</td><td>0.853126</td></tr><tr><td>9999999</td><td>0.892428</td></tr></tbody></table></div>




```python
%%timeit
df.uuid.with_uuid4()
```

    644 ms ± 6.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    
