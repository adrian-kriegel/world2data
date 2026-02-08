#!/usr/bin/env python
"""World2Data Human-in-the-Loop Review UI (Prototype).

A Gradio-based web interface for reviewing and correcting AI-detected
scene objects. This is the human review layer that allows domain experts
to verify, correct, and annotate the pipeline's output.

The expert can:
  - View each detected object with its confidence score
  - Correct labels and types
  - Mark objects as verified
  - Add notes
  - Flag false positives for removal
  - Save corrections back to the review JSON AND to 90_overrides.usda

Usage:
    uv run world2data-review --json data/outputs/demo_output/demo_human_review.json
    uv run world2data-review --json data/outputs/demo_output/demo_human_review.json --port 7861
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path


def load_review_data(json_path: str) -> dict:
    """Load the human review JSON."""
    with open(json_path, "r") as f:
        return json.load(f)


def save_review_data(data: dict, json_path: str):
    """Save corrected review data back to disk."""
    data["last_reviewed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _write_overrides_usda(objects: list[dict], output_dir: str):
    """Write a 90_overrides.usda file from human corrections.

    This connects the review UI to the OpenUSD layering protocol.
    Only objects with actual human changes are written as overrides.
    """
    try:
        from world2data.usd_layers import USDLayerWriter, OverrideRecord
    except ImportError:
        return None

    scene_dir = os.path.join(output_dir, "scene")
    if not os.path.isdir(scene_dir):
        # No layered scene exists, skip
        return None

    overrides = []
    for obj in objects:
        if not obj.get("human_verified") and not obj.get("human_label") and not obj.get("false_positive"):
            continue  # No human edits on this object

        entity_safe = obj.get("entity", "unknown").replace(" ", "_").replace("/", "_")
        prim_path = f"/World/W2D/Entities/Objects/{entity_safe}"

        if obj.get("human_label"):
            overrides.append(OverrideRecord(
                target_prim_path=prim_path,
                property_name="w2d:label",
                new_value=obj["human_label"],
                reason=obj.get("human_notes", "Human correction"),
                approved_by="human_reviewer",
            ))
        if obj.get("human_type") and obj["human_type"] != "(keep current)":
            overrides.append(OverrideRecord(
                target_prim_path=prim_path,
                property_name="w2d:class",
                new_value=obj["human_type"],
                reason=obj.get("human_notes", "Human correction"),
                approved_by="human_reviewer",
            ))
        if obj.get("false_positive"):
            overrides.append(OverrideRecord(
                target_prim_path=prim_path,
                property_name="w2d:falsePositive",
                new_value=True,
                reason=obj.get("human_notes", "Flagged as false positive"),
                approved_by="human_reviewer",
            ))
        if obj.get("human_verified"):
            overrides.append(OverrideRecord(
                target_prim_path=prim_path,
                property_name="w2d:humanVerified",
                new_value=True,
                reason="Human verified",
                approved_by="human_reviewer",
            ))

    if not overrides:
        return None

    writer = USDLayerWriter(scene_dir)
    override_path = writer.write_overrides_layer(overrides)
    return str(override_path)


def build_ui(json_path: str, port: int = 7860):
    """Build and launch the Gradio review interface."""
    try:
        import gradio as gr
    except ImportError:
        print("ERROR: gradio not installed. Run: uv add gradio")
        sys.exit(1)

    review_data = load_review_data(json_path)
    objects = review_data.get("objects", [])
    interactions = review_data.get("interactions", [])
    narrative = review_data.get("scene_narrative", "")

    # Determine output dir (for writing overrides USDA)
    output_dir = str(Path(json_path).parent)

    # =====================================================================
    # Build the Gradio app
    # =====================================================================

    # We collect component references to read values on save
    component_refs = []  # list of (obj_index, label, type, joint, verified, fp, notes)

    with gr.Blocks(
        title="World2Data - Human Review",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("# World2Data -- Human-in-the-Loop Review")
        gr.Markdown(
            f"**Video:** `{review_data.get('video_path', 'unknown')}`  \n"
            f"**Generated:** {review_data.get('generated_at', 'unknown')}  \n"
            f"**Objects:** {len(objects)} | "
            f"**Interactions:** {len(interactions)} | "
            f"**Needs Review:** "
            f"{sum(1 for o in objects if o.get('needs_review'))} objects"
        )

        # Scene narrative
        if narrative:
            with gr.Accordion("Scene Narrative (from Gemini Video Analysis)", open=False):
                gr.Markdown(narrative)

        # =================================================================
        # Objects tab
        # =================================================================
        with gr.Tab("Objects"):
            gr.Markdown("### Review Detected Objects")
            gr.Markdown(
                "Objects flagged with **Needs Review** have low confidence "
                "or were detected by only one model. Correct labels, verify, "
                "or flag as false positive."
            )

            # Summary table
            table_data = []
            for obj in objects:
                status = "NEEDS REVIEW" if obj.get("needs_review") else "OK"
                if obj.get("human_verified"):
                    status = "VERIFIED"
                table_data.append([
                    obj.get("id", 0),
                    obj.get("entity", "?"),
                    obj.get("type", "?"),
                    f"{obj.get('confidence', 0):.0%}",
                    ", ".join(obj.get("detected_by", [])),
                    obj.get("component_type", "?"),
                    status,
                ])

            gr.Dataframe(
                value=table_data,
                headers=["ID", "Entity", "Type", "Confidence",
                         "Detected By", "Joint Type", "Status"],
                label="All Detected Objects",
            )

            # Per-object review cards
            for idx, obj in enumerate(objects):
                needs_review = obj.get("needs_review", False)
                confidence = obj.get("confidence", 0)

                with gr.Accordion(
                    f"{'[!] ' if needs_review else ''}"
                    f"{obj.get('entity', '?')} ({obj.get('type', '?')}) "
                    f"-- {confidence:.0%} confidence",
                    open=needs_review,
                ):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown(
                                f"**Entity:** {obj.get('entity')}  \n"
                                f"**Type:** {obj.get('type')}  \n"
                                f"**Joint:** {obj.get('component_type')}  \n"
                                f"**State:** {obj.get('initial_state')} "
                                f"-> {obj.get('final_state')}  \n"
                                f"**Detected by:** "
                                f"{', '.join(obj.get('detected_by', []))}  \n"
                                f"**3D Center:** "
                                f"{[round(x, 2) for x in obj.get('center_3d', [0,0,0])]}  \n"
                                f"**3D Size:** "
                                f"{[round(x, 2) for x in obj.get('size_3d', [0,0,0])]}"
                            )
                        with gr.Column(scale=2):
                            human_label = gr.Textbox(
                                label="Correct Label (leave empty if correct)",
                                value=obj.get("human_label") or "",
                                placeholder="e.g., 'Coffee Table' if 'table' is too vague",
                            )
                            human_type = gr.Dropdown(
                                label="Correct Type",
                                choices=["(keep current)", "table", "chair", "door",
                                         "cup", "bottle", "person", "window",
                                         "cabinet", "shelf", "tv", "laptop",
                                         "phone", "book", "lamp", "sofa",
                                         "bed", "sink", "toilet", "other"],
                                value="(keep current)",
                            )
                            human_joint = gr.Dropdown(
                                label="Correct Joint Type",
                                choices=["(keep current)", "FixedJoint",
                                         "RevoluteJoint", "PrismaticJoint",
                                         "SphericalJoint", "FreeJoint"],
                                value="(keep current)",
                            )
                            human_verified = gr.Checkbox(
                                label="Mark as Verified",
                                value=obj.get("human_verified", False),
                            )
                            false_positive = gr.Checkbox(
                                label="Flag as False Positive (remove)",
                                value=False,
                            )
                            human_notes = gr.Textbox(
                                label="Notes",
                                value=obj.get("human_notes", ""),
                                placeholder="Any observations...",
                            )

                            component_refs.append(
                                (idx, human_label, human_type, human_joint,
                                 human_verified, false_positive, human_notes)
                            )

        # =================================================================
        # Interactions tab
        # =================================================================
        with gr.Tab("Interactions"):
            gr.Markdown("### State Changes & Interactions")

            if interactions:
                interaction_table = []
                for inter in interactions:
                    interaction_table.append([
                        inter.get("entity", "?"),
                        inter.get("time", "?"),
                        inter.get("from_state", "?"),
                        inter.get("to_state", "?"),
                        inter.get("cause", "?"),
                        "Yes" if inter.get("human_verified") else "No",
                    ])

                gr.Dataframe(
                    value=interaction_table,
                    headers=["Entity", "Time", "From", "To", "Cause", "Verified"],
                    label="Detected Interactions",
                )
            else:
                gr.Markdown("*No interactions detected.*")

        # =================================================================
        # Evaluation tab
        # =================================================================
        with gr.Tab("Quality Metrics"):
            gr.Markdown("### Pipeline Quality Evaluation")

            n_objects = len(objects)
            n_review = sum(1 for o in objects if o.get("needs_review"))
            n_verified = sum(1 for o in objects if o.get("human_verified"))
            avg_conf = (
                sum(o.get("confidence", 0) for o in objects) / max(n_objects, 1)
            )
            multi_model = sum(
                1 for o in objects if len(o.get("detected_by", [])) >= 2
            )

            gr.Markdown(
                f"| Metric | Value |\n"
                f"|--------|-------|\n"
                f"| Total objects | {n_objects} |\n"
                f"| Needs review | {n_review} |\n"
                f"| Human verified | {n_verified} |\n"
                f"| Avg confidence | {avg_conf:.0%} |\n"
                f"| Multi-model agreement | {multi_model}/{n_objects} |\n"
                f"| Interactions | {len(interactions)} |\n"
            )

        # =================================================================
        # USD Viewer tab
        # =================================================================
        with gr.Tab("USD Scene"):
            gr.Markdown("### OpenUSD Scene Hierarchy")
            gr.Markdown(
                "The USD file contains the full 3D scene with all objects, "
                "physics metadata, and accuracy scores.\n\n"
                "To view interactively:\n"
                "```bash\n"
                "usdview demo_output/scene/scene.usda\n"
                "```\n"
                "Or open in NVIDIA Omniverse."
            )

            # Show USD tree from the JSON sidecar
            json_sidecar = json_path.replace("_human_review.json",
                                              "_scene_graph.json")
            if os.path.isfile(json_sidecar):
                with open(json_sidecar) as f:
                    sidecar = json.load(f)

                tree_lines = ["/World"]
                n_pts = sidecar.get("num_points", 0)
                tree_lines.append(f"  /PointCloud ({n_pts:,} points)")
                n_cams = sidecar.get("num_cameras", 0)
                tree_lines.append(f"  /Cameras ({n_cams} cameras)")
                tree_lines.append(f"  /Objects")
                for obj in sidecar.get("objects_3d", []):
                    entity = obj.get("entity", "?")
                    otype = obj.get("type", "?")
                    conf = obj.get("tracking_confidence",
                                   obj.get("confidence", "?"))
                    joint = obj.get("component_type", "?")
                    tree_lines.append(
                        f"    /{entity} ({otype}) "
                        f"[{joint}] conf={conf}"
                    )

                gr.Code(
                    value="\n".join(tree_lines),
                    label="USD Scene Tree",
                    language=None,
                )

        # =================================================================
        # Save button
        # =================================================================
        save_btn = gr.Button(
            "Save All Changes (JSON + USD Overrides)",
            variant="primary", size="lg",
        )
        save_status = gr.Markdown("")

        # Build the list of all input components for the save handler
        all_inputs = []
        for (idx, label, typ, joint, verified, fp, notes) in component_refs:
            all_inputs.extend([label, typ, joint, verified, fp, notes])

        def save_changes(*values):
            """Read all component values and update the review JSON + USD overrides."""
            n_components_per_obj = 6  # label, type, joint, verified, fp, notes
            changes_count = 0

            for i, (obj_idx, *_) in enumerate(component_refs):
                offset = i * n_components_per_obj
                label_val = values[offset]
                type_val = values[offset + 1]
                joint_val = values[offset + 2]
                verified_val = values[offset + 3]
                fp_val = values[offset + 4]
                notes_val = values[offset + 5]

                obj = objects[obj_idx]
                changed = False

                if label_val and label_val.strip():
                    obj["human_label"] = label_val.strip()
                    changed = True
                if type_val and type_val != "(keep current)":
                    obj["human_type"] = type_val
                    changed = True
                if joint_val and joint_val != "(keep current)":
                    obj["human_joint"] = joint_val
                    changed = True
                if verified_val:
                    obj["human_verified"] = True
                    changed = True
                if fp_val:
                    obj["false_positive"] = True
                    changed = True
                if notes_val and notes_val.strip():
                    obj["human_notes"] = notes_val.strip()
                    changed = True

                if changed:
                    changes_count += 1

            # Save updated JSON
            review_data["objects"] = objects
            save_review_data(review_data, json_path)

            # Write 90_overrides.usda if layered scene exists
            override_path = _write_overrides_usda(objects, output_dir)
            override_msg = ""
            if override_path:
                override_msg = f"  \nUSD overrides written to `{override_path}`"

            return (
                f"Saved {changes_count} changes to `{json_path}` "
                f"at {time.strftime('%H:%M:%S')}.{override_msg}"
            )

        save_btn.click(fn=save_changes, inputs=all_inputs, outputs=save_status)

    # Launch
    print(f"\nLaunching World2Data Review UI on http://localhost:{port}")
    print(f"Review file: {json_path}")
    app.launch(server_port=port, share=False, inbrowser=True)


def main():
    parser = argparse.ArgumentParser(
        description="World2Data Human Review UI"
    )
    parser.add_argument("--json", required=True,
                        help="Path to human review JSON file")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port for Gradio server (default 7860)")
    args = parser.parse_args()

    if not os.path.isfile(args.json):
        print(f"ERROR: Review JSON not found: {args.json}")
        sys.exit(1)

    build_ui(args.json, args.port)


if __name__ == "__main__":
    main()
