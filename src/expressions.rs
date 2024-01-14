use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use std::fmt::Write;
use uuid::Uuid;

fn uuid4(_value: &str, output: &mut String) {
    let id = Uuid::new_v4();
    write!(output, "{}",id.braced()).unwrap()
}

#[polars_expr(output_type=String)]
fn create_uuid4(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    match s.dtype() {
        DataType::String => {
            let ca =inputs[0].str()?;
            let out = ca.apply_to_buffer(uuid4);
            Ok(out.into_series())
        }
        _ =>{
            let binding = inputs[0].cast(&DataType::String)?;
            let ca = binding.str()?;
            let out = ca.apply_to_buffer(uuid4);
            Ok(out.into_series())
        }
        
    }
}