use std::ffi::{c_double, c_uint, c_void};
use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use std::mem;
use std::panic::catch_unwind;
use std::ptr::{copy_nonoverlapping, null_mut};
use crate::{Element, Matrix};

/// A more C-friendly format for `Matrix<T>` that must be 
/// manually deallocated.
pub struct CompactMatrix<T>
where T: Element<T>
{
    pub rows: usize,
    pub cols: usize,
    pub vals: *mut T,
}

impl <T> CompactMatrix<T>
where T: Element<T>
{
    /// Converts a `CompactMatrix<T>` to a `Matrix<T>`, allowing
    /// one to operate on a pointer passed from another language.
    /// 
    /// NOTE: This does not own or deallocate the data in a 
    /// `*mut CompactMatrix<T>`, it must be manually `dealloc`ed 
    /// to prevent leaking memory.
    fn to_matrix(&self) -> Matrix<T>
    {
        let nvals = self.rows * self.cols;
        Matrix
        {
            rows: self.rows,
            cols: self.cols,
            vals: unsafe { Vec::from_raw_parts(self.vals, nvals, nvals) },
        }
    }
}

#[no_mangle]
pub extern "C" fn new_double_matrix(rows: c_uint, cols: c_uint) -> *mut c_void
{
    let res = catch_unwind(|| {
        let a = Matrix::new(rows as usize, cols as usize).to_compact_matrix();
        let ptr: *mut CompactMatrix<c_double>;
        let layout = Layout::new::<CompactMatrix<c_double>>();
        unsafe
        {
            ptr = alloc(layout) as *mut CompactMatrix<c_double>;
            if ptr.is_null() 
            {
                handle_alloc_error(layout);
            }
            copy_nonoverlapping(&a as *const CompactMatrix<c_double>, ptr, 1);
        }
        ptr as *mut c_void
    });
    
    match res
    {
        Ok(ptr) => ptr,
        Err(_)  => null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn new_double_identity_matrix(n: c_uint) -> *mut c_void
{
    let res = catch_unwind(|| {
        let a = Matrix::new_identity(n as usize).to_compact_matrix();
        let ptr: *mut CompactMatrix<c_double>;
        let layout = Layout::new::<CompactMatrix<c_double>>();
        unsafe
        {
            ptr = alloc(layout) as *mut CompactMatrix<c_double>;
            if ptr.is_null() 
            {
                handle_alloc_error(layout); 
            }
            copy_nonoverlapping(&a as *const CompactMatrix<c_double>, ptr, 1);
        }
        ptr as *mut c_void
    });
    
    match res
    {
        Ok(ptr) => ptr,
        Err(_)  => null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn inplace_row_swap(ptr: *mut c_void, r1: c_uint, r2: c_uint) -> c_uint
{
    let res = catch_unwind(|| {
        let mut a = unsafe 
        { 
            (*(ptr as *mut CompactMatrix<c_double>)).to_matrix() 
        };
        a.inplace_row_swap(r1 as usize, r2 as usize);
        
        mem::forget(a); // Prevent drop that would deallocate the matrix data
    });

    match res
    {
        Ok(_)  => 1,
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "C" fn inplace_row_scale(ptr: *mut c_void, row: c_uint, scalar: c_double) -> c_uint
{
    let res = catch_unwind(|| {
        let mut a = unsafe 
        { 
            (*(ptr as *mut CompactMatrix<c_double>)).to_matrix() 
        };
        a.inplace_row_scale(row as usize, scalar);

        mem::forget(a); // Prevent drop that would deallocate the matrix data
    });

    match res
    {
        Ok(_)  => 1,
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "C" fn inplace_scale(ptr: *mut c_void, scalar: c_double) -> c_uint
{
    let res = catch_unwind(|| {
        let mut a = unsafe 
        {
            (*(ptr as *mut CompactMatrix<c_double>)).to_matrix()
        };
        a.inplace_scale(scalar);

        mem::forget(a); // Prevent drop that would deallocate the matrix data
    });

    match res
    {
        Ok(_)  => 1,
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "C" fn inplace_row_add(ptr: *mut c_void, r1: c_uint, r2: c_uint) -> c_uint
{
    let res = catch_unwind(|| {
        let mut a = unsafe 
        { 
            (*(ptr as *mut CompactMatrix<c_double>)).to_matrix() 
        };
        a.inplace_row_add(r1 as usize, r2 as usize);

        mem::forget(a); // Prevent drop that would deallocate the matrix data
    });

    match res
    {
        Ok(_)  => 1,
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "C" fn inplace_scaled_row_add(ptr: *mut c_void, r1: c_uint, r2: c_uint, scalar: c_double) -> c_uint
{
    let res = catch_unwind(|| {
        let mut a = unsafe 
        { 
            (*(ptr as *mut CompactMatrix<c_double>)).to_matrix()
        };
        a.inplace_scaled_row_add(r1 as usize, r2 as usize, scalar);

        mem::forget(a); // Prevent drop that would deallocate the matrix data
    });

    match res
    {
        Ok(_)  => 1,
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "C" fn multiply_matrix(ptr_a: *mut c_void, ptr_b: *mut c_void) -> *mut c_void
{
    let res = catch_unwind(|| {
        let (a, b) = unsafe 
        {(
            (*(ptr_a as *mut CompactMatrix<c_double>)).to_matrix(),
            (*(ptr_b as *mut CompactMatrix<c_double>)).to_matrix(),
        )};
    
        let ab = match a.multiply_matrix(&b) 
        {
            Ok(x)  => x.to_compact_matrix(),
            Err(_) => return null_mut(), // return early and indicate failure via NULL
        };
    
        mem::forget(a); // Prevent drop that would deallocate matrix data. We don't inform the
        mem::forget(b); // caller that a or b will be deallocated, so we shouldn't do it here.
    
        let ptr: *mut CompactMatrix<c_double>;
        let layout = Layout::new::<CompactMatrix<c_double>>();
    
        unsafe 
        {
            ptr = alloc(layout) as *mut CompactMatrix<c_double>;
            if ptr.is_null()
            { 
                handle_alloc_error(layout);
            }
            copy_nonoverlapping(&ab, ptr, 1);
        }
        
        ptr as *mut c_void
    });

    match res
    {
        Ok(ptr) => ptr,
        Err(_)  => null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn augment_with(ptr_a: *mut c_void, ptr_b: *mut c_void) -> *mut c_void
{
    let res = catch_unwind(|| {
        let (a, b) = unsafe 
        {(
            (*(ptr_a as *mut CompactMatrix<c_double>)).to_matrix(),
            (*(ptr_b as *mut CompactMatrix<c_double>)).to_matrix(),
        )};
    
        let ab = match a.augment_with(&b) 
        {
            Ok(x)  => x.to_compact_matrix(),
            Err(_) => return null_mut(), // return early and indicate failure via NULL
        };
    
        mem::forget(a); // Prevent drop that would deallocate matrix data. We don't inform the
        mem::forget(b); // caller that a or b will be deallocated, so we shouldn't do it here.
    
        let ptr: *mut CompactMatrix<c_double>;
        let layout = Layout::new::<CompactMatrix<c_double>>();
    
        unsafe 
        {
            ptr = alloc(layout) as *mut CompactMatrix<c_double>;
            if ptr.is_null() 
            { 
                handle_alloc_error(layout); 
            }
            copy_nonoverlapping(&ab as *const CompactMatrix<c_double>, ptr, 1);
        }
        
        ptr as *mut c_void
    });
    
    match res
    {
        Ok(ptr) => ptr,
        Err(_)  => null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn subset(ptr: *mut c_void, r1: c_uint, c1: c_uint, r2: c_uint, c2: c_uint) -> *mut c_void
{
    let res = catch_unwind(|| {
        let a = unsafe
        { 
            (*(ptr as *mut CompactMatrix<c_double>)).to_matrix() 
        };

        let b = a.subset(r1 as usize, c1 as usize, r2 as usize, c2 as usize).to_compact_matrix();
        
        mem::forget(a); // Prevent drop that would deallocate the matrix data

        let newptr: *mut CompactMatrix<c_double>;
        let layout = Layout::new::<CompactMatrix<c_double>>();

        unsafe 
        {
            newptr = alloc(layout) as *mut CompactMatrix<c_double>;
            if newptr.is_null()
            {
                handle_alloc_error(layout);
            }
            copy_nonoverlapping(&b, newptr, 1);
        }

        newptr as *mut c_void
    });

    match res
    {
        Ok(ptr) => ptr,
        Err(_)  => null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn trace(ptr: *mut c_void) -> c_double
{
    let res = catch_unwind(|| {
        let a = unsafe 
        { 
            (*(ptr as *mut CompactMatrix<c_double>)).to_matrix() 
        };
        let trace = match a.trace()
        {
            Ok(t) => t,
            Err(_) => c_double::MIN
        };

        mem::forget(a); // Prevent drop that would deallocate the matrix data

        trace
    });

    match res 
    {
        Ok(t)  => t as c_double,
        Err(_) => c_double::MIN,
    }
}

#[no_mangle]
pub extern "C" fn transpose(ptr: *mut c_void) -> *mut c_void
{
    let res = catch_unwind(|| {
        let a = unsafe
        {
            (*(ptr as *mut CompactMatrix<c_double>)).to_matrix()
        };

        let b = a.transpose();
        mem::forget(a);

        let newptr: *mut CompactMatrix<c_double>;
        let layout = Layout::new::<CompactMatrix<c_double>>();

        unsafe
        {
            newptr = alloc(layout) as *mut CompactMatrix<c_double>;
            if ptr.is_null() 
            {
                handle_alloc_error(layout);
            }
            copy_nonoverlapping(&b.to_compact_matrix() as *const CompactMatrix<c_double>, newptr, 1);
        }

        newptr
    });

    match res
    {
        Ok(ptr) => ptr as *mut c_void,
        Err(_)  => null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn try_inplace_invert(ptr: *mut c_void) -> c_uint
{
    let res = catch_unwind(|| {
        let mut a = unsafe 
        {
            (*(ptr as *mut CompactMatrix<c_double>)).to_matrix()
        };
    
        let status = match a.try_inplace_invert()
        {
            Ok(_)  => 1,
            Err(_) => 0,
        };
    
        mem::forget(a);
        status
    });
    
    match res
    {
        Ok(stat) => stat,
        Err(_)   => 0,
    }
}

#[no_mangle]
pub extern "C" fn index_mut_double_matrix(ptr: *mut c_void, i: c_uint, j: c_uint, value: c_double) -> c_uint
{
    let res = catch_unwind(|| {
        let mut a = unsafe 
        {
            (*(ptr as *mut CompactMatrix<c_double>)).to_matrix()
        };
        a[(i as usize, j as usize)] = value;

        mem::forget(a);
    });

    match res
    {
        Ok(_)  => 1,
        Err(_) => 0,
    }
}

#[no_mangle]
pub extern "C" fn index_double_matrix(ptr: *mut c_void, i: c_uint, j: c_uint) -> c_double
{
    let res = catch_unwind(|| {
        let a = unsafe 
        {
            (*(ptr as *mut CompactMatrix<c_double>)).to_matrix()
        };
        let value = a[(i as usize, j as usize)];

        mem::forget(a);

        value
    });

    match res
    {
        Ok(o)  => o,
        Err(_) =>
        {
            c_double::MIN
        }
    }
}

#[no_mangle]
pub extern "C" fn clone_double_matrix(ptr: *mut c_void) -> *mut c_void
{
    let res = catch_unwind(|| {
        let newptr: *mut CompactMatrix<c_double>;
        let layout = Layout::new::<Matrix<c_double>>();
        let a = unsafe
        {
            // Get the actual matrix instance
            (*(ptr as *mut CompactMatrix<c_double>)).to_matrix()
        };

        // Use clone to allocate a new instance and mem::forget it AND the old instance
        let b = a.clone().to_compact_matrix();
        mem::forget(a);

        unsafe 
        {
            newptr = alloc(layout) as *mut CompactMatrix<c_double>;
            if newptr.is_null()
            {
                handle_alloc_error(layout); 
            }
            copy_nonoverlapping(&b, newptr, 1);
        };
    
        newptr as *mut c_void
    });
    
    match res
    {
        Ok(ptr) => ptr,
        Err(_)  => null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn free_double_matrix(ptr: *mut c_void) -> ()
{
    // Try to dealloc. if a panic occurs, abort and leak mem 
    // to avoid UB in the name of Ferris.
    let _ = catch_unwind(|| {
        let layout = Layout::new::<CompactMatrix<c_double>>();
        unsafe 
        {
            // Own the data in the pointer to drop the matrix data
            (*(ptr as *mut CompactMatrix<c_double>)).to_matrix();
            // Vec<T> w/ Matrix data goes out of scope...
    
            dealloc(ptr as *mut u8, layout);
            // void* with metadata is manually dealloced here...
        }
    });
}
