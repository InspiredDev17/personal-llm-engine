//! Isolated C ABI ownership and unsafe pointer boundary.
#![allow(unsafe_code)]

use personalai_engine_types::EngineError;
use std::ffi::{c_char, c_void};
use std::sync::Arc;

#[repr(C)]
struct Context {
    _private: [u8; 0],
}
#[repr(C)]
struct Buffer {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct RawParams {
    pub count: u32,
    pub rows: u32,
    pub columns: u32,
    pub inner: u32,
    pub position: u32,
    pub head_dimension: u32,
    pub head_count: u32,
    pub kv_head_count: u32,
    pub sequence_length: u32,
    pub query_count: u32,
    pub first_query_position: u32,
    pub block_count: u32,
    pub scalar0: f32,
    pub scalar1: f32,
    pub scalar2: f32,
    pub scalar3: f32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct RawDeviceInfo {
    registry_id: u64,
    recommended: u64,
    maximum: u64,
    max_threads: u32,
    unified: u8,
    apple7: u8,
    f16: u8,
    reserved: u8,
    name: [c_char; 128],
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawStatistics {
    current: u64,
    peak: u64,
    allocations: u64,
    h2d: u64,
    d2h: u64,
    dispatches: u64,
    failures: u64,
}

unsafe extern "C" {
    fn pai_metal_device_count() -> usize;
    fn pai_metal_device_info(index: usize, out: *mut RawDeviceInfo) -> i32;
    fn pai_metal_context_create(
        index: usize,
        bytes: *const u8,
        length: usize,
        out: *mut *mut Context,
    ) -> i32;
    fn pai_metal_context_destroy(context: *mut Context);
    fn pai_metal_buffer_create(
        context: *mut Context,
        length: usize,
        alignment: usize,
        mode: i32,
        out: *mut *mut Buffer,
    ) -> i32;
    fn pai_metal_buffer_destroy(buffer: *mut Buffer);
    fn pai_metal_buffer_length(buffer: *const Buffer) -> usize;
    fn pai_metal_upload(
        context: *mut Context,
        source: *const u8,
        length: usize,
        destination: *mut Buffer,
    ) -> i32;
    fn pai_metal_download(
        context: *mut Context,
        source: *const Buffer,
        destination: *mut u8,
        length: usize,
    ) -> i32;
    fn pai_metal_upload_range(
        context: *mut Context,
        source: *const u8,
        offset: usize,
        length: usize,
        destination: *mut Buffer,
    ) -> i32;
    fn pai_metal_download_range(
        context: *mut Context,
        source: *const Buffer,
        offset: usize,
        destination: *mut u8,
        length: usize,
    ) -> i32;
    fn pai_metal_buffer_clear(context: *mut Context, buffer: *mut Buffer) -> i32;
    fn pai_metal_dispatch(
        context: *mut Context,
        kernel: i32,
        buffers: *const *mut Buffer,
        offsets: *const usize,
        count: usize,
        parameters: *const RawParams,
        gx: u32,
        gy: u32,
        gz: u32,
    ) -> i32;
    fn pai_metal_synchronize(context: *mut Context) -> i32;
    fn pai_metal_statistics(context: *mut Context, out: *mut RawStatistics) -> i32;
    fn pai_metal_last_error(
        context: *mut Context,
        destination: *mut c_char,
        capacity: usize,
    ) -> usize;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    pub registry_id: u64,
    pub recommended_bytes: u64,
    pub maximum_buffer_bytes: u64,
    pub maximum_threads: u32,
    pub unified_memory: bool,
    pub apple7: bool,
    pub f16: bool,
    pub name: String,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Statistics {
    pub current_bytes: u64,
    pub peak_bytes: u64,
    pub allocations: u64,
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
    pub dispatches: u64,
    pub command_failures: u64,
}

#[derive(Debug)]
pub struct NativeContext(usize);
impl Drop for NativeContext {
    fn drop(&mut self) {
        unsafe {
            pai_metal_context_destroy(self.0 as *mut Context);
        }
    }
}
#[derive(Debug)]
pub struct NativeBuffer {
    context: Arc<NativeContext>,
    handle: usize,
    length: usize,
}
impl Drop for NativeBuffer {
    fn drop(&mut self) {
        unsafe {
            pai_metal_buffer_destroy(self.handle as *mut Buffer);
        }
    }
}

pub fn device_count() -> usize {
    unsafe { pai_metal_device_count() }
}
pub fn device_info(index: usize) -> Result<DeviceInfo, EngineError> {
    let mut raw = RawDeviceInfo {
        registry_id: 0,
        recommended: 0,
        maximum: 0,
        max_threads: 0,
        unified: 0,
        apple7: 0,
        f16: 0,
        reserved: 0,
        name: [0; 128],
    };
    status(None, unsafe { pai_metal_device_info(index, &raw mut raw) })?;
    let length = raw.name.iter().position(|v| *v == 0).unwrap_or(raw.name.len());
    let bytes = raw.name[..length].iter().map(|value| value.to_ne_bytes()[0]).collect::<Vec<_>>();
    let name = String::from_utf8(bytes)
        .map_err(|_| EngineError::backend_failure("Metal device name is invalid UTF-8"))?;
    Ok(DeviceInfo {
        registry_id: raw.registry_id,
        recommended_bytes: raw.recommended,
        maximum_buffer_bytes: raw.maximum,
        maximum_threads: raw.max_threads,
        unified_memory: raw.unified != 0,
        apple7: raw.apple7 != 0,
        f16: raw.f16 != 0,
        name,
    })
}
pub fn create_context(index: usize, library: &[u8]) -> Result<Arc<NativeContext>, EngineError> {
    let mut raw = std::ptr::null_mut();
    status(None, unsafe {
        pai_metal_context_create(index, library.as_ptr(), library.len(), &raw mut raw)
    })?;
    if raw.is_null() {
        return Err(EngineError::backend_failure("Metal context creation returned null"));
    }
    Ok(Arc::new(NativeContext(raw as usize)))
}
impl NativeContext {
    fn pointer(&self) -> *mut Context {
        self.0 as *mut Context
    }
    pub fn allocate(
        self: &Arc<Self>,
        length: usize,
        alignment: usize,
        private: bool,
    ) -> Result<NativeBuffer, EngineError> {
        let mut raw = std::ptr::null_mut();
        status(Some(self), unsafe {
            pai_metal_buffer_create(
                self.pointer(),
                length,
                alignment,
                i32::from(private),
                &raw mut raw,
            )
        })?;
        if raw.is_null() {
            return Err(EngineError::backend_failure("Metal buffer creation returned null"));
        }
        let actual = unsafe { pai_metal_buffer_length(raw) };
        if actual != length {
            unsafe { pai_metal_buffer_destroy(raw) };
            return Err(EngineError::backend_failure("Metal buffer length differs"));
        }
        Ok(NativeBuffer { context: Arc::clone(self), handle: raw as usize, length })
    }
    pub fn synchronize(&self) -> Result<(), EngineError> {
        status(Some(self), unsafe { pai_metal_synchronize(self.pointer()) })
    }
    pub fn statistics(&self) -> Result<Statistics, EngineError> {
        let mut raw = RawStatistics::default();
        status(Some(self), unsafe { pai_metal_statistics(self.pointer(), &raw mut raw) })?;
        Ok(Statistics {
            current_bytes: raw.current,
            peak_bytes: raw.peak,
            allocations: raw.allocations,
            host_to_device_bytes: raw.h2d,
            device_to_host_bytes: raw.d2h,
            dispatches: raw.dispatches,
            command_failures: raw.failures,
        })
    }
    pub fn dispatch(
        &self,
        kernel: i32,
        buffers: &[&NativeBuffer],
        params: &RawParams,
        grid: [u32; 3],
    ) -> Result<(), EngineError> {
        let handles = buffers.iter().map(|b| b.handle as *mut Buffer).collect::<Vec<_>>();
        let offsets = vec![0usize; buffers.len()];
        status(Some(self), unsafe {
            pai_metal_dispatch(
                self.pointer(),
                kernel,
                handles.as_ptr(),
                offsets.as_ptr(),
                handles.len(),
                params,
                grid[0],
                grid[1],
                grid[2],
            )
        })
    }
}
impl NativeBuffer {
    pub const fn len(&self) -> usize {
        self.length
    }
    pub fn upload(&self, bytes: &[u8]) -> Result<(), EngineError> {
        status(Some(&self.context), unsafe {
            pai_metal_upload(
                self.context.pointer(),
                bytes.as_ptr(),
                bytes.len(),
                self.handle as *mut Buffer,
            )
        })
    }
    pub fn download(&self) -> Result<Vec<u8>, EngineError> {
        let mut bytes = vec![0u8; self.length];
        status(Some(&self.context), unsafe {
            pai_metal_download(
                self.context.pointer(),
                self.handle as *const Buffer,
                bytes.as_mut_ptr(),
                bytes.len(),
            )
        })?;
        Ok(bytes)
    }
    pub fn upload_range(&self, offset: usize, bytes: &[u8]) -> Result<(), EngineError> {
        status(Some(&self.context), unsafe {
            pai_metal_upload_range(
                self.context.pointer(),
                bytes.as_ptr(),
                offset,
                bytes.len(),
                self.handle as *mut Buffer,
            )
        })
    }
    pub fn download_range(&self, offset: usize, length: usize) -> Result<Vec<u8>, EngineError> {
        let mut bytes = vec![0; length];
        status(Some(&self.context), unsafe {
            pai_metal_download_range(
                self.context.pointer(),
                self.handle as *const Buffer,
                offset,
                bytes.as_mut_ptr(),
                length,
            )
        })?;
        Ok(bytes)
    }
    pub fn clear(&self) -> Result<(), EngineError> {
        status(Some(&self.context), unsafe {
            pai_metal_buffer_clear(self.context.pointer(), self.handle as *mut Buffer)
        })
    }
}
fn status(context: Option<&NativeContext>, code: i32) -> Result<(), EngineError> {
    if code == 0 {
        return Ok(());
    }
    let detail = context
        .map_or_else(|| "Metal native boundary rejected the operation".to_owned(), last_error);
    match code {
        1 => Err(EngineError::invalid_input(detail)),
        2 => Err(EngineError::not_implemented(detail)),
        3 => Err(EngineError::resource_exhaustion(detail)),
        _ => Err(EngineError::backend_failure(detail)),
    }
}
fn last_error(context: &NativeContext) -> String {
    let length =
        unsafe { pai_metal_last_error(context.pointer(), std::ptr::null_mut(), 0) }.min(512);
    if length == 0 {
        return "Metal native operation failed".to_owned();
    }
    let mut bytes = vec![0u8; length + 1];
    unsafe { pai_metal_last_error(context.pointer(), bytes.as_mut_ptr().cast(), bytes.len()) };
    bytes.truncate(length);
    String::from_utf8(bytes).unwrap_or_else(|_| "Metal native error text was invalid".to_owned())
}

#[allow(dead_code)]
fn _opaque(_: *mut c_void) {}
