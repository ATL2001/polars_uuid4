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




    '0.20.3'



##### Make dataframe of with 10 million random numbers


```python
df = pl.DataFrame({
    'Random numbers': np.random.rand(10000000),
    'A string column': "value",
}).with_row_count()
df.tail()
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (5, 3)</small><table border="1" class="dataframe"><thead><tr><th>row_nr</th><th>Random numbers</th><th>A string column</th></tr><tr><td>u32</td><td>f64</td><td>str</td></tr></thead><tbody><tr><td>9999995</td><td>0.410216</td><td>&quot;value&quot;</td></tr><tr><td>9999996</td><td>0.072977</td><td>&quot;value&quot;</td></tr><tr><td>9999997</td><td>0.763713</td><td>&quot;value&quot;</td></tr><tr><td>9999998</td><td>0.536438</td><td>&quot;value&quot;</td></tr><tr><td>9999999</td><td>0.703031</td><td>&quot;value&quot;</td></tr></tbody></table></div>



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
<small>shape: (10_000_000, 4)</small><table border="1" class="dataframe"><thead><tr><th>row_nr</th><th>Random numbers</th><th>A string column</th><th>uuid</th></tr><tr><td>u32</td><td>f64</td><td>str</td><td>str</td></tr></thead><tbody><tr><td>0</td><td>0.758339</td><td>&quot;value&quot;</td><td>&quot;{3952aa21-0957…</td></tr><tr><td>1</td><td>0.04649</td><td>&quot;value&quot;</td><td>&quot;{0708f057-7e56…</td></tr><tr><td>2</td><td>0.498708</td><td>&quot;value&quot;</td><td>&quot;{e655242c-cad8…</td></tr><tr><td>3</td><td>0.726538</td><td>&quot;value&quot;</td><td>&quot;{dc0d153c-71bd…</td></tr><tr><td>4</td><td>0.161975</td><td>&quot;value&quot;</td><td>&quot;{efef8d80-b8d0…</td></tr><tr><td>5</td><td>0.391948</td><td>&quot;value&quot;</td><td>&quot;{6b8f3261-3554…</td></tr><tr><td>6</td><td>0.341304</td><td>&quot;value&quot;</td><td>&quot;{5b1b6c85-96dd…</td></tr><tr><td>7</td><td>0.965395</td><td>&quot;value&quot;</td><td>&quot;{414e16d9-5c73…</td></tr><tr><td>8</td><td>0.689368</td><td>&quot;value&quot;</td><td>&quot;{ccbacf05-2857…</td></tr><tr><td>9</td><td>0.628131</td><td>&quot;value&quot;</td><td>&quot;{cb402ac3-3bfe…</td></tr><tr><td>10</td><td>0.643579</td><td>&quot;value&quot;</td><td>&quot;{5ba78915-fff1…</td></tr><tr><td>11</td><td>0.939639</td><td>&quot;value&quot;</td><td>&quot;{262364fb-534d…</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>9999988</td><td>0.512794</td><td>&quot;value&quot;</td><td>&quot;{c67f2d3b-fd65…</td></tr><tr><td>9999989</td><td>0.108904</td><td>&quot;value&quot;</td><td>&quot;{7107720a-9b12…</td></tr><tr><td>9999990</td><td>0.834744</td><td>&quot;value&quot;</td><td>&quot;{8447f2f4-b763…</td></tr><tr><td>9999991</td><td>0.987605</td><td>&quot;value&quot;</td><td>&quot;{aae4b490-ba50…</td></tr><tr><td>9999992</td><td>0.973912</td><td>&quot;value&quot;</td><td>&quot;{aa698d5c-6970…</td></tr><tr><td>9999993</td><td>0.82106</td><td>&quot;value&quot;</td><td>&quot;{59cf6971-aae2…</td></tr><tr><td>9999994</td><td>0.080472</td><td>&quot;value&quot;</td><td>&quot;{c5d4edc6-68d4…</td></tr><tr><td>9999995</td><td>0.410216</td><td>&quot;value&quot;</td><td>&quot;{e6619727-4d97…</td></tr><tr><td>9999996</td><td>0.072977</td><td>&quot;value&quot;</td><td>&quot;{dc162d63-8bee…</td></tr><tr><td>9999997</td><td>0.763713</td><td>&quot;value&quot;</td><td>&quot;{361df98d-259e…</td></tr><tr><td>9999998</td><td>0.536438</td><td>&quot;value&quot;</td><td>&quot;{fd538ddf-7b46…</td></tr><tr><td>9999999</td><td>0.703031</td><td>&quot;value&quot;</td><td>&quot;{5fc43590-f343…</td></tr></tbody></table></div>



#### Works with a lazy frame too


```python
df = pl.LazyFrame({
    'Random numbers': np.random.rand(10000000),
    'A string column': "value",
}).with_row_count().uuid.with_uuid4().collect()
df.tail()
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (5, 4)</small><table border="1" class="dataframe"><thead><tr><th>row_nr</th><th>Random numbers</th><th>A string column</th><th>uuid</th></tr><tr><td>u32</td><td>f64</td><td>str</td><td>str</td></tr></thead><tbody><tr><td>9999995</td><td>0.313736</td><td>&quot;value&quot;</td><td>&quot;{a7a4d264-da44…</td></tr><tr><td>9999996</td><td>0.833383</td><td>&quot;value&quot;</td><td>&quot;{71fd5366-c708…</td></tr><tr><td>9999997</td><td>0.5506</td><td>&quot;value&quot;</td><td>&quot;{294ec201-2df2…</td></tr><tr><td>9999998</td><td>0.339044</td><td>&quot;value&quot;</td><td>&quot;{2fbde60d-a46e…</td></tr><tr><td>9999999</td><td>0.896245</td><td>&quot;value&quot;</td><td>&quot;{1673458a-565e…</td></tr></tbody></table></div>



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

    20.4 s ± 225 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    

##### Using pl_uuid to generate a UUID4 for each row 
  * Gets job done.  Creates a UUID4 for each row.
  * Uses rust uuid crate.
  * Much easier to understand/simpler code.
  * ~ 40x faster than using python's uuid module to generate UUID4 when the last column in the df is already a string
  * 540 ms ± 6.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


```python
%%timeit
df.uuid.with_uuid4()
```

    540 ms ± 6.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    

##### Not quite as fast if there isnt an existing string column in the dataframe
  * 656 ms ± 6.66 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


```python
df = pl.DataFrame({
    'Random numbers': np.random.rand(10000000),
}).with_row_count()
df.tail()
```




<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<small>shape: (5, 2)</small><table border="1" class="dataframe"><thead><tr><th>row_nr</th><th>Random numbers</th></tr><tr><td>u32</td><td>f64</td></tr></thead><tbody><tr><td>9999995</td><td>0.334474</td></tr><tr><td>9999996</td><td>0.006089</td></tr><tr><td>9999997</td><td>0.295484</td></tr><tr><td>9999998</td><td>0.065312</td></tr><tr><td>9999999</td><td>0.644697</td></tr></tbody></table></div>




```python
%%timeit
df.uuid.with_uuid4()
```

    656 ms ± 6.66 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    


```python

```
