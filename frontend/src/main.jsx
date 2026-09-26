import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";

import "./index.css";
import Layout from "./components/Layout";
import Home from "./pages/Home";
import StockDetail from "./pages/StockDetail";
import Research from "./pages/Research";
import Engineering from "./pages/Engineering";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Home />} />
          <Route path="/stock/:ticker" element={<StockDetail />} />
          <Route path="/evidence" element={<Research />} />
          <Route path="/engineering" element={<Engineering />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);
