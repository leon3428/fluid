use std::time;

use winit::{application::ApplicationHandler, dpi::PhysicalSize, event::WindowEvent, event_loop::{ActiveEventLoop, ControlFlow, EventLoop}, window::{Window, WindowId}};

use crate::application_state::State;

pub struct App {
    state: Option<State>,
    last_frame_time: time::Instant,
}

impl App {
    pub fn new() -> Self {
        Self { state: None, last_frame_time: time::Instant::now() }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(Window::default_attributes().with_inner_size(PhysicalSize::new(800, 800)))
            .unwrap();

        self.state = Some(State::new(window));
        self.last_frame_time = time::Instant::now();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let window = self.state.as_ref().unwrap().window();

        if window.id() == window_id {
            match event {
                WindowEvent::CloseRequested => {
                    event_loop.exit();
                }
                WindowEvent::RedrawRequested => {
                    let now = time::Instant::now();
                    let delta_time = now.duration_since(self.last_frame_time);
                    self.last_frame_time = now;
                    let delta_seconds = delta_time.as_secs_f32();

                    let state = self.state.as_mut().unwrap();
                    state.update(delta_seconds);
                    state.render().unwrap();
                }
                _ => {}
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let window = self.state.as_ref().unwrap().window();
        window.request_redraw();
    }
}

pub fn run() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    let _ = event_loop.run_app(&mut app);
}