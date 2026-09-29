---
schema_version: '1.0'
id: kg:reference:triton-blocked-programming-model
kind: reference
title: Triton uses a blocked-program execution model
summary: Structure Triton kernels around program instances that each process blocks
  of values; program_id identifies the current instance.
claim_key: reference.triton.blocked_programming_model
domains:
- language
- programming-model
status: stable
sources:
- resource: source:triton-programming-guide
  locator: blocked-programming-model
  title: Triton Programming Guide - Introduction
scope:
  target:
    level: portable
    software:
      language: triton
  numerics:
    exact: false
retrieval:
  phases:
  - initial
  tasks:
  - architecture_selection
  - implementation
  techniques:
  - blocked_program
  keywords:
  - triton
  - program_id
  - grid
evidence_state: source_supported
managed:
  content_hash: sha256:bb8202d213a252dce61741127794666ef6196712bd7c1aa93146a951158f4439
  created_by: kernelgen/seed-v1
  created_at: '2026-07-26T00:00:00Z'
  updated_at: '2026-07-26T00:00:00Z'
---

# Fact

Triton launches a grid of program instances. Each instance operates on blocks
of values, and `program_id(axis)` selects its position in that grid.

# Use

Choose grid dimensions and block ownership before selecting low-level tuning
parameters. This fact does not prescribe a specific tile size.
