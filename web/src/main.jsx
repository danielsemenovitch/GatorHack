import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
// Self-hosted Archivo (width + weight axes) so the design renders offline at a venue.
import "@fontsource-variable/archivo/wdth.css";
import "./styles.css";

createRoot(document.getElementById("root")).render(<App />);
