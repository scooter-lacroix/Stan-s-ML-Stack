# Rust Style Guide

A comprehensive guide for writing idiomatic, safe, and performant Rust code. This guide follows the official Rust style guidelines, community best practices, and modern Rust development patterns (2025/2026).

## Table of Contents

- [Code Formatting](#code-formatting)
- [Naming Conventions](#naming-conventions)
- [Ownership and Borrowing](#ownership-and-borrowing)
- [Error Handling](#error-handling)
- [Type System](#type-system)
- [Functions and Methods](#functions-and-methods)
- [Structs and Enums](#structs-and-enums)
- [Traits and Generics](#traits-and-generics)
- [Concurrency](#concurrency)
- [Unsafe Rust](#unsafe-rust)
- [Tooling and Configuration](#tooling-and-configuration)
- [Common Patterns](#common-patterns)
- [Anti-Patterns to Avoid](#anti-patterns-to-avoid)

---

## Code Formatting

### Using `rustfmt`

**Always run `rustfmt`** on your code before committing. Rust has an official formatting tool that enforces consistent style.

```bash
# Format your code
cargo fmt

# Check if formatting is needed
cargo fmt --check
```

### Indentation and Line Length

- Use **4 spaces** for indentation (no tabs)
- Maximum line length: **100 characters** (soft limit)
- `rustfmt` handles most formatting automatically

```rust
// Good: Proper indentation
fn process_data(input: &str) -> Result<String, Error> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(Error::EmptyInput);
    }
    Ok(trimmed.to_uppercase())
}

// Good: Long lines are wrapped automatically
fn very_long_function_name(
    parameter_one: &str,
    parameter_two: &str,
    parameter_three: &str,
) -> Result<String, Error> {
    // ...
}
```

### Braces and Whitespace

```rust
// Good: Opening brace on same line, closing brace on new line
fn main() {
    let x = 42;
    if x > 0 {
        println!("Positive");
    }
}

// Good: Space after colon in type annotations
let mut name: String = String::new();
let numbers: Vec<i32> = Vec::new();

// Good: Spaces around operators
let sum = a + b;
let result = x * y + z;

// Good: No trailing whitespace
let valid = true;
```

---

## Naming Conventions

### General Rules

| Category | Convention | Example |
|----------|-----------|---------|
| Variables, Functions, Methods | `snake_case` | `user_name`, `calculate_total` |
| Structs, Enums, Traits | `PascalCase` | `UserService`, `HttpError` |
| Constants | `SCREAMING_SNAKE_CASE` | `MAX_CONNECTIONS`, `DEFAULT_TIMEOUT` |
| Lifetime Parameters | Short, lowercase | `'a`, `'de`, `'src` |
| Generic Types | Short, PascalCase | `T`, `E`, `K`, `V` |
| Modules | `snake_case` | `user_service`, `http_client` |

### Detailed Examples

```rust
// Good: Variable and function naming
let user_count = 42;
let is_active = true;
let max_retries = 3;

fn calculate_total(items: &[Item]) -> f64 {
    items.iter().map(|i| i.price).sum()
}

fn get_user_by_id(id: u32) -> Option<User> {
    // ...
}

// Good: Struct and enum naming
struct UserProfile {
    id: u32,
    name: String,
    email: String,
}

enum HttpStatus {
    Ok,
    NotFound,
    ServerError,
}

// Good: Trait naming
trait Printable {
    fn print(&self);
}

trait Iterator {
    type Item;
    fn next(&mut self) -> Option<Self::Item>;
}

// Good: Constant naming
const MAX_CONNECTIONS: usize = 100;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
const API_BASE_URL: &str = "https://api.example.com";

// Good: Module naming
mod user_service;
mod http_client;
mod database;

// Good: Acronyms are treated as words
// Use "Http" not "HTTP", "Json" not "JSON"
struct HttpResponse {
    // ...
}

fn parse_json(input: &str) -> Result<Value, Error> {
    // ...
}
```

### Boolean Naming

```rust
// Good: Prefix booleans with 'is', 'has', 'should', 'can'
let is_valid = true;
let has_permission = false;
let should_retry = true;
let can_delete = false;

// Good: Descriptive predicate names
fn is_empty(&self) -> bool {
    self.len() == 0
}

fn has_permission(&self, permission: &str) -> bool {
    self.permissions.contains(permission)
}

fn should_retry(&self, attempt: u32) -> bool {
    attempt < self.max_attempts
}
```

---

## Ownership and Borrowing

### Understanding Ownership

```rust
// Good: Ownership transfer (move)
fn process_user(user: User) -> String {
    // user is now owned by this function
    format!("User: {}", user.name)
} // user is dropped here

// Good: Borrowing - immutable reference
fn print_user(user: &User) {
    println!("User: {}", user.name);
} // No drop, only borrowed

// Good: Borrowing - mutable reference
fn rename_user(user: &mut User, new_name: String) {
    user.name = new_name;
}
```

### Borrowing Rules

```rust
// Good: Multiple immutable references
let x = 42;
let r1 = &x;
let r2 = &x;
let r3 = &x;
println!("{} {} {}", r1, r2, r3);

// Good: Single mutable reference
let mut x = 42;
let r1 = &mut x;
*r1 += 1;
println!("{}", r1);

// Bad: Cannot have mutable and immutable references simultaneously
// let mut x = 42;
// let r1 = &x;
// let r2 = &mut x; // Error!
// println!("{}", r1);
```

### Lifetime Annotations

```rust
// Good: Explicit lifetime annotations
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

// Good: Lifetime elision rules apply when obvious
fn first_word(s: &str) -> &str {
    // Lifetime is elided (compiler infers it)
    match s.find(' ') {
        Some(index) => &s[..index],
        None => s,
    }
}

// Good: Structs with lifetime annotations
struct Context<'a> {
    data: &'a str,
    metadata: &'a Metadata,
}

impl<'a> Context<'a> {
    fn new(data: &'a str, metadata: &'a Metadata) -> Self {
        Context { data, metadata }
    }
}

// Good: 'static lifetime for string literals
const API_URL: &'static str = "https://api.example.com";

fn get_config() -> &'static str {
    "production"
}
```

### Clone vs Copy

```rust
// Good: Use Copy for cheap, simple types
#[derive(Debug, Copy, Clone)]
struct Point {
    x: i32,
    y: i32,
}

// Good: Implement Clone for types that need deep copying
#[derive(Debug, Clone)]
struct User {
    id: u32,
    name: String,
    email: String,
}

// Good: Explicit clone when you need a copy
fn process_user(user: &User) -> User {
    let mut user_copy = user.clone();
    user_copy.name = user_copy.name.to_uppercase();
    user_copy
}

// Good: Borrow instead of cloning when possible
fn print_user_name(user: &User) {
    println!("{}", user.name); // No clone needed
}
```

---

## Error Handling

### Result Type

```rust
// Good: Use Result for fallible operations
fn parse_user_id(input: &str) -> Result<u32, ParseError> {
    match input.trim().parse::<u32>() {
        Ok(id) => Ok(id),
        Err(_) => Err(ParseError::InvalidFormat),
    }
}

// Good: Use ? operator for propagation
fn read_user_file(path: &Path) -> Result<String, io::Error> {
    let content = fs::read_to_string(path)?; // Propagates error
    Ok(content)
}

// Good: Chain operations with ?
fn process_user_file(path: &Path) -> Result<User, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let user: User = serde_json::from_str(&content)?;
    Ok(user)
}
```

### Custom Error Types

```rust
// Good: Define custom error types
#[derive(Debug)]
enum AppError {
    Io(io::Error),
    Parse(ParseError),
    Network(reqwest::Error),
    Custom(String),
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppError::Io(e) => write!(f, "IO error: {}", e),
            AppError::Parse(e) => write!(f, "Parse error: {:?}", e),
            AppError::Network(e) => write!(f, "Network error: {}", e),
            AppError::Custom(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::error::Error for AppError {}

// Good: Implement From for automatic conversions
impl From<io::Error> for AppError {
    fn from(err: io::Error) -> Self {
        AppError::Io(err)
    }
}

impl From<ParseError> for AppError {
    fn from(err: ParseError) -> Self {
        AppError::Parse(err)
    }
}

// Usage with ? operator
fn read_config(path: &Path) -> Result<Config, AppError> {
    let content = fs::read_to_string(path)?; // Converts io::Error to AppError
    let config: Config = serde_json::from_str(&content)?;
    Ok(config)
}
```

### Using thiserror and anyhow

```rust
// Good: Use thiserror for custom errors in libraries
use thiserror::Error;

#[derive(Error, Debug)]
pub enum UserServiceError {
    #[error("User not found: {id}")]
    NotFound { id: u32 },

    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

// Good: Use anyhow for application-level error handling
use anyhow::{Context, Result};

async fn fetch_user(id: u32) -> Result<User> {
    let user = db::find_by_id(id)
        .await
        .context("Failed to fetch user from database")?;
    Ok(user)
}
```

### Option vs Result

```rust
// Good: Use Option for values that may or may not exist
fn get_user_by_id(id: u32, users: &[User]) -> Option<&User> {
    users.iter().find(|u| u.id == id)
}

// Good: Use Result for operations that can fail
fn parse_user_id(input: &str) -> Result<u32, ParseIntError> {
    input.parse::<u32>()
}

// Good: Convert Option to Result with context
fn get_user_required(id: u32, users: &[User]) -> Result<&User, String> {
    get_user_by_id(id, users).ok_or_else(|| format!("User {} not found", id))
}

// Good: Use combinators
let user = get_user_by_id(id, users)
    .ok_or(UserServiceError::NotFound { id })?;
```

---

## Type System

### Primitive Types

```rust
// Good: Use appropriate integer types
let count: u32 = 42;         // Unsigned, 32-bit
let index: usize = 0;        // Size-specific (for indexing)
let temperature: i32 = -5;   // Signed, 32-bit
let byte: u8 = 255;          // Unsigned, 8-bit

// Good: Use f64 for floating point (default)
let pi: f64 = 3.14159;

// Good: Use bool for boolean values
let is_active: bool = true;

// Good: Use char for Unicode scalar values
let c: char = 'A';

// Good: Use &str for string slices (borrowed)
let greeting: &str = "Hello, World!";

// Good: Use String for owned strings
let mut message: String = String::from("Hello");
message.push_str(", World!");
```

### Collections

```rust
// Good: Use Vec for growable arrays
let mut numbers: Vec<i32> = Vec::new();
numbers.push(1);
numbers.push(2);
let first = numbers[0];

// Good: Use vec! macro for initialization
let numbers = vec![1, 2, 3, 4, 5];

// Good: Use HashMap for key-value pairs
use std::collections::HashMap;

let mut scores: HashMap<String, i32> = HashMap::new();
scores.insert(String::from("Alice"), 10);
scores.insert(String::from("Bob"), 20);

// Good: Use HashSet for unique values
use std::collections::HashSet;

let mut unique_numbers: HashSet<i32> = HashSet::new();
unique_numbers.insert(1);
unique_numbers.insert(2);
unique_numbers.insert(1); // Duplicate, ignored

// Good: Use tuples for heterogeneous data
let tuple: (i32, f64, &str) = (42, 3.14, "hello");
let (x, y, z) = tuple;
```

### Smart Pointers

```rust
// Good: Use Box for heap allocation
let large_data = Box::new([0u8; 1024]);

// Good: Use Box for recursive types
enum List {
    Cons(i32, Box<List>),
    Nil,
}

// Good: Use Rc for shared ownership
use std::rc::Rc;

let shared_data = Rc::new(vec![1, 2, 3]);
let another_reference = Rc::clone(&shared_data);

// Good: Use Arc for thread-safe shared ownership
use std::sync::Arc;

let shared_data = Arc::new(vec![1, 2, 3]);
let thread_data = Arc::clone(&shared_data);

// Good: Use RefCell for interior mutability
use std::cell::RefCell;

let data = RefCell::new(vec![1, 2, 3]);
*data.borrow_mut() += 1;
```

---

## Functions and Methods

### Function Signatures

```rust
// Good: Descriptive function names
fn calculate_total_price(items: &[CartItem]) -> f64 {
    items.iter().map(|item| item.price * item.quantity as f64).sum()
}

// Good: Use reference parameters for large data
fn process_users(users: &[User]) -> Vec<UserSummary> {
    users.iter().map(|u| UserSummary::from(u)).collect()
}

// Good: Return Result for fallible operations
fn read_config(path: &Path) -> Result<Config, io::Error> {
    let content = fs::read_to_string(path)?;
    let config: Config = toml::from_str(&content)?;
    Ok(config)
}

// Good: Use Option for nullable returns
fn find_user_by_email(email: &str, users: &[User]) -> Option<&User> {
    users.iter().find(|u| u.email == email)
}
```

### Method Definitions

```rust
// Good: Implement methods with impl blocks
impl User {
    // Good: Constructor-like function (new)
    pub fn new(name: String, email: String) -> Self {
        User {
            id: generate_id(),
            name,
            email,
            created_at: Utc::now(),
        }
    }

    // Good: Borrowed self for read-only operations
    pub fn display_name(&self) -> String {
        format!("{} ({})", self.name, self.email)
    }

    // Good: Mutable self for modifying operations
    pub fn rename(&mut self, new_name: String) {
        self.name = new_name;
        self.updated_at = Utc::now();
    }

    // Good: Consuming self for transformations
    pub fn into_admin(self) -> AdminUser {
        AdminUser {
            id: self.id,
            name: self.name,
            permissions: vec![],
        }
    }
}
```

### Closure Best Practices

```rust
// Good: Use closures for short, simple operations
let numbers = vec![1, 2, 3, 4, 5];
let doubled: Vec<i32> = numbers.iter().map(|x| x * 2).collect();

// Good: Use move when closure needs ownership
let data = vec![1, 2, 3];
std::thread::spawn(move || {
    println!("Data: {:?}", data);
}).join().unwrap();

// Good: Use type annotations for complex closures
let mut cache = HashMap::new();
let mut get_or_insert = |key: &str| -> &i32 {
    cache.entry(key.to_string()).or_insert(0)
};

// Good: Use Fn, FnMut, FnOnce traits for generics
fn apply_twice<F>(mut f: F, value: i32) -> i32
where
    F: FnMut(i32) -> i32,
{
    f(f(value))
}

let result = apply_twice(|x| x + 1, 5); // Returns 7
```

---

## Structs and Enums

### Struct Definitions

```rust
// Good: Field structs for data aggregation
struct User {
    id: u32,
    name: String,
    email: String,
    created_at: DateTime<Utc>,
}

// Good: Tuple structs for newtypes
struct UserId(u32);
struct Email(String);

impl UserId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn value(&self) -> u32 {
        self.0
    }
}

// Good: Unit structs for type markers
struct Secure;
struct Public;

// Good: Use derives for common traits
#[derive(Debug, Clone, PartialEq, Eq)]
struct User {
    id: u32,
    name: String,
}

// Good: Builder pattern for complex construction
struct UserBuilder {
    name: Option<String>,
    email: Option<String>,
    age: Option<u32>,
}

impl UserBuilder {
    pub fn new() -> Self {
        UserBuilder {
            name: None,
            email: None,
            age: None,
        }
    }

    pub fn name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    pub fn email(mut self, email: String) -> Self {
        self.email = Some(email);
        self
    }

    pub fn age(mut self, age: u32) -> Self {
        self.age = Some(age);
        self
    }

    pub fn build(self) -> Result<User, String> {
        Ok(User {
            id: generate_id(),
            name: self.name.ok_or("name is required")?,
            email: self.email.ok_or("email is required")?,
            age: self.age.unwrap_or(0),
        })
    }
}

// Usage
let user = UserBuilder::new()
    .name("John".to_string())
    .email("john@example.com".to_string())
    .age(30)
    .build()?;
```

### Enum Definitions

```rust
// Good: Simple enums for fixed choices
enum Direction {
    North,
    South,
    East,
    West,
}

// Good: Enums with data (C-like or tuple variants)
enum IpAddr {
    V4(u8, u8, u8, u8),
    V6(String),
}

// Good: Enums with struct variants
enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(i32, i32, i32),
}

// Good: Option-like enums for optional values
enum MaybeUser {
    User(User),
    Guest,
}

// Good: Result-like enums for error handling
enum ApiResult<T> {
    Success(T),
    Error { code: u32, message: String },
}

// Good: Use derives for enums
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum HttpStatus {
    Ok = 200,
    NotFound = 404,
    ServerError = 500,
}

// Good: Implement methods on enums
impl HttpStatus {
    pub fn is_success(self) -> bool {
        (200..300).contains(&(self as u32))
    }

    pub fn is_error(self) -> bool {
        (400..600).contains(&(self as u32))
    }
}
```

### Pattern Matching

```rust
// Good: Exhaustive matching on enums
fn describe_message(msg: Message) {
    match msg {
        Message::Quit => println!("Quit"),
        Message::Move { x, y } => println!("Move to ({}, {})", x, y),
        Message::Write(text) => println!("Write: {}", text),
        Message::ChangeColor(r, g, b) => println!("Color: ({}, {}, {})", r, g, b),
    }
}

// Good: Use _ for catch-all
fn describe_direction(direction: Direction) {
    match direction {
        Direction::North => println!("North"),
        Direction::South => println!("South"),
        _ => println!("Other direction"),
    }
}

// Good: Destructuring in match
fn get_user_name(user: Option<User>) -> String {
    match user {
        Some(User { name, .. }) => name,
        None => "Unknown".to_string(),
    }
}

// Good: if let for single pattern matching
if let Some(user) = get_user(42) {
    println!("Found user: {}", user.name);
}

// Good: while let for repeated pattern matching
let mut iter = numbers.iter();
while let Some(value) = iter.next() {
    println!("{}", value);
}

// Good: let else for early returns
fn parse_config(path: &Path) -> Result<Config, Error> {
    let Some(extension) = path.extension() else {
        return Err(Error::MissingExtension);
    };

    let content = fs::read_to_string(path)?;
    // ...
}
```

---

## Traits and Generics

### Trait Definitions

```rust
// Good: Define traits for shared behavior
trait Printable {
    fn print(&self);
    fn print_details(&self) {
        println!("Default implementation");
    }
}

// Good: Implement traits for types
impl Printable for User {
    fn print(&self) {
        println!("User: {}", self.name);
    }
}

// Good: Trait with associated types
trait Iterator {
    type Item;
    fn next(&mut self) -> Option<Self::Item>;
}

// Good: Trait bounds for generics
fn debug_print<T: std::fmt::Debug>(item: T) {
    println!("{:?}", item);
}

// Good: Multiple trait bounds
fn compare<T: PartialEq + Clone>(a: T, b: T) -> bool {
    a == b
}

// Good: where clause for complex bounds
fn process<T, U>(t: T, u: U) -> String
where
    T: std::fmt::Display,
    U: std::fmt::Display,
{
    format!("{} and {}", t, u)
}
```

### Derive Macros

```rust
// Good: Use derives for common traits
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Point {
    x: i32,
    y: i32,
}

// Good: Custom derives with serde
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
struct User {
    id: u32,
    name: String,
    email: String,
}

// Good: Default derive
#[derive(Default, Debug)]
struct Config {
    host: String,
    port: u16,
    timeout: u64,
}

// Good: Implement Default manually
impl Default for Config {
    fn default() -> Self {
        Config {
            host: "localhost".to_string(),
            port: 8080,
            timeout: 30,
        }
    }
}
```

### Generic Best Practices

```rust
// Good: Use generics for reusable code
struct Repository<T> {
    items: Vec<T>,
}

impl<T> Repository<T> {
    pub fn new() -> Self {
        Repository { items: Vec::new() }
    }

    pub fn add(&mut self, item: T) {
        self.items.push(item);
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.items.get(index)
    }
}

// Good: Constrain generics with traits
fn print_all<T: std::fmt::Display>(items: &[T]) {
    for item in items {
        println!("{}", item);
    }
}

// Good: Use lifetime parameters with generics
struct Context<'a, T> {
    data: &'a T,
    metadata: &'a Metadata,
}

// Good: Associated types for trait generics
trait Container {
    type Item;
    fn get(&self, index: usize) -> Option<&Self::Item>;
}
```

---

## Concurrency

### Threads

```rust
// Good: Use std::thread for basic concurrency
use std::thread;
use std::time::Duration;

thread::spawn(|| {
    for i in 1..10 {
        println!("Hi number {} from the spawned thread!", i);
        thread::sleep(Duration::from_millis(1));
    }
});

// Good: Move closures for ownership
let data = vec![1, 2, 3, 4, 5];
thread::spawn(move || {
    println!("Data: {:?}", data);
});

// Good: Join handles for waiting
let handle = thread::spawn(|| {
    42
});

let result = handle.join().unwrap(); // Returns 42

// Good: Use scoped threads for borrowing
use crossbeam::scope;

let data = vec![1, 2, 3, 4, 5];

scope(|s| {
    s.spawn(|| {
        println!("First thread: {:?}", data);
    });

    s.spawn(|| {
        println!("Second thread: {:?}", data);
    });
}).unwrap();
```

### Channels

```rust
// Good: Use channels for message passing
use std::sync::mpsc;

let (tx, rx) = mpsc::channel();

thread::spawn(move || {
    let val = String::from("hi");
    tx.send(val).unwrap();
});

let received = rx.recv().unwrap();

// Good: Multiple producers
let (tx, rx) = mpsc::channel();
let tx1 = tx.clone();
let tx2 = tx.clone();

thread::spawn(move || {
    tx1.send(1).unwrap();
});

thread::spawn(move || {
    tx2.send(2).unwrap();
});

for received in rx {
    println!("Got: {}", received);
}

// Good: Use sync channels for bounded communication
let (tx, rx) = mpsc::sync_channel(5);

for i in 0..10 {
    tx.send(i).unwrap();
    println!("Sent {}", i);
}
```

### Shared State

```rust
// Good: Use Mutex for interior mutability
use std::sync::{Arc, Mutex};

let counter = Arc::new(Mutex::new(0));
let mut handles = vec![];

for _ in 0..10 {
    let counter = Arc::clone(&counter);
    let handle = thread::spawn(move || {
        let mut num = counter.lock().unwrap();
        *num += 1;
    });
    handles.push(handle);
}

for handle in handles {
    handle.join().unwrap();
}

println!("Result: {}", *counter.lock().unwrap());

// Good: Use RwLock for read-heavy workloads
use std::sync::RwLock;

let data = Arc::new(RwLock::new(vec![1, 2, 3, 4, 5]));

// Many readers
{
    let r1 = data.read().unwrap();
    let r2 = data.read().unwrap();
    println!("{:?}, {:?}", r1, r2);
} // Read locks released here

// Single writer
{
    let mut w = data.write().unwrap();
    w.push(6);
} // Write lock released here
```

### Async Rust

```rust
// Good: Use async/await for asynchronous code
use tokio::time::{sleep, Duration};

async fn hello_world() {
    println!("Hello");
    sleep(Duration::from_secs(1)).await;
    println!("World");
}

// Good: Async functions return futures
async fn fetch_user(id: u32) -> Result<User, Error> {
    let response = reqwest::get(format!("/users/{}", id)).await?;
    let user = response.json().await?;
    Ok(user)
}

// Good: Use tokio::main for async runtime
#[tokio::main]
async fn main() -> Result<(), Error> {
    let user = fetch_user(42).await?;
    println!("User: {:?}", user);
    Ok(())
}

// Good: Join for concurrent operations
let (user, posts) = tokio::join!(
    fetch_user(42),
    fetch_posts(42)
);

// Good: Try_join for fallible concurrent operations
use tokio::try_join;

let (user, posts) = try_join!(
    fetch_user(42),
    fetch_posts(42)
)?;
```

---

## Unsafe Rust

### When to Use Unsafe

```rust
// Good: Document why unsafe is needed
/// # Safety
/// This function is safe to call when `ptr` is valid and aligned,
/// and points to memory that can be read as a u32.
unsafe fn read_u32(ptr: *const u32) -> u32 {
    *ptr
}

// Good: Use unsafe in small, well-documented blocks
fn safe_abstraction(ptr: *const u32) -> Option<u32> {
    if ptr.is_null() {
        return None;
    }
    // Safety: We checked for null, and assuming pointer is valid
    Some(unsafe { *ptr })
}

// Good: Implementing custom smart pointers
struct MyBox<T> {
    ptr: *mut T,
}

impl<T> MyBox<T> {
    fn new(value: T) -> Self {
        let ptr = Box::into_raw(Box::new(value));
        MyBox { ptr }
    }
}

impl<T> Drop for MyBox<T> {
    fn drop(&mut self) {
        // Safety: ptr was created with Box::into_raw
        unsafe {
            drop(Box::from_raw(self.ptr));
        }
    }
}

impl<T> std::ops::Deref for MyBox<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // Safety: ptr is valid and aligned
        unsafe { &*self.ptr }
    }
}
```

---

## Tooling and Configuration

### Cargo.toml Best Practices

```toml
[package]
name = "my_project"
version = "0.1.0"
edition = "2021"
rust-version = "1.70"
authors = ["Your Name <you@example.com>"]
description = "A brief description"
license = "MIT OR Apache-2.0"
repository = "https://github.com/yourusername/my_project"
keywords = ["cli", "tool"]
categories = ["command-line-utilities"]

[dependencies]
# Good: Pin dependency versions
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.35", features = ["full"] }
anyhow = "1.0"
thiserror = "1.0"

# Good: Use path dependencies for workspace
local_package = { path = "../local_package" }

[dev-dependencies]
criterion = "0.5"
proptest = "1.4"

[profile.release]
# Good: Optimize for size
opt-level = "z"     # Optimize for size
lto = true          # Link-time optimization
codegen-units = 1   # Better optimization
strip = true        # Remove debug symbols

[profile.dev]
# Good: Faster compilation for development
opt-level = 0

[[bin]]
name = "my_cli"
path = "src/main.rs"
```

### Clippy

```bash
# Run clippy for additional lints
cargo clippy

# Run clippy with all features
cargo clippy --all-features

# Fix clippy warnings automatically
cargo clippy --fix
```

### Testing

```rust
// Good: Unit tests in the same module
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_addition() {
        assert_eq!(add(2, 2), 4);
    }

    #[test]
    fn test_panic() {
        #[should_panic(expected = "Division by zero")]
        divide(1, 0);
    }

    #[test]
    fn test_result() {
        assert!(matches!(parse_number("42"), Ok(42)));
    }
}

// Good: Integration tests in tests/ directory
// tests/integration_test.rs
#[test]
fn test_full_workflow() {
    // Test the full application workflow
}

// Good: Use property-based testing
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_add_commutative(a in any::<i32>(), b in any::<i32>()) {
        prop_assert_eq!(add(a, b), add(b, a));
    }
}
```

---

## Common Patterns

### Newtype Pattern

```rust
// Good: Wrapper types for type safety
struct UserId(u32);
struct Email(String);

impl UserId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn value(&self) -> u32 {
        self.0
    }
}

fn get_user(id: UserId) -> Option<User> {
    // ...
}

// Usage prevents mixing up IDs
let user_id = UserId::new(42);
get_user(user_id);
```

### Builder Pattern (Revisited)

```rust
// Good: Use build patterns for complex construction
pub struct HttpRequest {
    url: String,
    method: String,
    headers: Vec<(String, String)>,
    body: Option<String>,
}

pub struct HttpRequestBuilder {
    url: Option<String>,
    method: Option<String>,
    headers: Vec<(String, String)>,
    body: Option<String>,
}

impl HttpRequestBuilder {
    pub fn new() -> Self {
        Self {
            url: None,
            method: None,
            headers: Vec::new(),
            body: None,
        }
    }

    pub fn url(mut self, url: String) -> Self {
        self.url = Some(url);
        self
    }

    pub fn method(mut self, method: String) -> Self {
        self.method = Some(method);
        self
    }

    pub fn header(mut self, key: String, value: String) -> Self {
        self.headers.push((key, value));
        self
    }

    pub fn body(mut self, body: String) -> Self {
        self.body = Some(body);
        self
    }

    pub fn build(self) -> Result<HttpRequest, String> {
        Ok(HttpRequest {
            url: self.url.ok_or("url is required")?,
            method: self.method.unwrap_or_else(|| "GET".to_string()),
            headers: self.headers,
            body: self.body,
        })
    }
}
```

### Strategy Pattern

```rust
// Good: Use trait objects for runtime polymorphism
trait Formatter {
    fn format(&self, data: &str) -> String;
}

struct JsonFormatter;
struct XmlFormatter;

impl Formatter for JsonFormatter {
    fn format(&self, data: &str) -> String {
        format!("{{\"data\": \"{}\"}}", data)
    }
}

impl Formatter for XmlFormatter {
    fn format(&self, data: &str) -> String {
        format!("<data>{}</data>", data)
    }
}

fn output_data(formatter: &dyn Formatter, data: &str) {
    println!("{}", formatter.format(data));
}
```

---

## Anti-Patterns to Avoid

### Don't Panic in Libraries

```rust
// Bad: Panicking in library code
fn divide(a: i32, b: i32) -> i32 {
    if b == 0 {
        panic!("Division by zero!");
    }
    a / b
}

// Good: Return Result
fn divide(a: i32, b: i32) -> Result<i32, String> {
    if b == 0 {
        return Err("Division by zero".to_string());
    }
    Ok(a / b)
}
```

### Don't Use Unnecessary Unsafe

```rust
// Bad: Unsafe when safe alternatives exist
unsafe fn get_first(vec: &Vec<i32>) -> Option<&i32> {
    if !vec.is_empty() {
        Some(&vec.get_unchecked(0))
    } else {
        None
    }
}

// Good: Use safe methods
fn get_first(vec: &Vec<i32>) -> Option<&i32> {
    vec.first()
}
```

### Don't Ignore Compiler Warnings

```rust
// Bad: Dead code produces warnings
#[allow(dead_code)]
fn unused_function() {
    // ...
}

// Good: Remove unused code or mark it appropriately
#[cfg(test)]
mod test_helpers {
    pub fn setup() {
        // ...
    }
}
```

---

## Additional Resources

- [The Rust Programming Language](https://doc.rust-lang.org/book/)
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/)
- [API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [Rust Std Documentation](https://doc.rust-lang.org/std/)
- [Rust Design Patterns](https://rust-unofficial.github.io/patterns/)
