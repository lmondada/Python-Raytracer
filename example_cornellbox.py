from sightpy import *


def main():
    # Set Scene
    Sc = Scene(ambient_color=rgb(0.00, 0.00, 0.00))
    angle = -0
    Sc.add_Camera(
        screen_width=100,
        screen_height=100,
        look_from=vec3(278, 278, 800),
        look_at=vec3(278, 278, 0),
        focal_distance=1.0,
        field_of_view=40,
    )

    # define materials to use
    green_diffuse = Diffuse(
        diff_color=rgb(0.33, 0.73, 0.43), diff_color_ref=rgb(0.33, 0.73, 0.43)
    )
    red_diffuse = Diffuse(
        diff_color=rgb(0.75, 0.25, 0.35), diff_color_ref=rgb(0.75, 0.25, 0.35)
    )
    white_diffuse = Diffuse(
        diff_color=rgb(0.73, 0.73, 0.73), diff_color_ref=rgb(0.73, 0.73, 0.73)
    )
    emissive_white = Emissive(color=rgb(0.9, 0.9, 0.9))
    blue_glass = Refractive(
        n=vec3(1.5 + 0.05e-8j, 1.5 + 0.02e-8j, 1.5 + 0.0j), purity=0.99, purity_ref=0.4
    )

    # this is the light
    Sc.add(
        Plane(
            material=emissive_white,
            center=vec3(213 + 130 / 2, 554, -227.0 - 105 / 2),
            width=130.0,
            height=105.0,
            u_axis=vec3(1.0, 0.0, 0),
            v_axis=vec3(0.0, 0, 1.0),
        ),
        importance_sampled=True,
    )

    Sc.add(
        Plane(
            material=white_diffuse,
            center=vec3(555 / 2, 555 / 2, -555.0),
            width=555.0,
            height=555.0,
            u_axis=vec3(0.0, 1.0, 0),
            v_axis=vec3(1.0, 0, 0.0),
        )
    )

    Sc.add(
        Plane(
            material=green_diffuse,
            center=vec3(-0.0, 555 / 2, -555 / 2),
            width=555.0,
            height=555.0,
            u_axis=vec3(0.0, 1.0, 0),
            v_axis=vec3(0.0, 0, -1.0),
        )
    )

    Sc.add(
        Plane(
            material=red_diffuse,
            center=vec3(555.0, 555 / 2, -555 / 2),
            width=555.0,
            height=555.0,
            u_axis=vec3(0.0, 1.0, 0),
            v_axis=vec3(0.0, 0, -1.0),
        )
    )

    Sc.add(
        Plane(
            material=white_diffuse,
            center=vec3(555 / 2, 555, -555 / 2),
            width=555.0,
            height=555.0,
            u_axis=vec3(1.0, 0.0, 0),
            v_axis=vec3(0.0, 0, -1.0),
        )
    )

    Sc.add(
        Plane(
            material=white_diffuse,
            center=vec3(555 / 2, 0.0, -555 / 2),
            width=555.0,
            height=555.0,
            u_axis=vec3(1.0, 0.0, 0),
            v_axis=vec3(0.0, 0, -1.0),
        )
    )

    cb = Cuboid(
        material=white_diffuse,
        center=vec3(182.5, 165, -285 - 160 / 2),
        width=165,
        height=165 * 2,
        length=165,
        shadow=False,
    )
    cb.rotate(θ=15, u=vec3(0, 1, 0))
    Sc.add(cb)

    Sc.add(
        Sphere(
            material=blue_glass,
            center=vec3(370.5, 165 / 2, -65 - 185 / 2),
            radius=165 / 2,
            shadow=False,
            max_ray_depth=3,
        ),
        importance_sampled=True,
    )
    # Render
    img, gold_bars = Sc.render(samples_per_pixel=48, progress_bar=True)

    img.save("cornell_box_1.png")
    # img.show()


if __name__ == "__main__":
    main()
